#include "gpu-patch.h"
#include "gpu-queue.h"
#include "utils.h"

#include <cub/cub.cuh>

#define GPU_ANALYSIS_DEBUG 0

#if GPU_ANALYSIS_DEBUG
#define PRINT(...) \
 if (threadIdx.x == 0 && blockIdx.x == 0) { \
   printf(__VA_ARGS__); \
 } 
#define PRINT_ALL(...) \
  printf(__VA_ARGS__)
#define PRINT_RECORDS(buffer) \
  __syncthreads(); \
  if (threadIdx.x == 0) { \
    gpu_patch_analysis_address_t *records = (gpu_patch_analysis_address_t *)buffer->records; \
    for (uint32_t i = 0; i < buffer->head_index; ++i) { \
      printf("gpu analysis-> merged <%p, %p> (%p)\n", records[i].start, records[i].end, records[i].end - records[i].start); \
    } \
  } \
  __syncthreads(); 
#else
#define PRINT(...)
#define PRINT_ALL(...)
#define PRINT_RECORDS(buffer) 
#endif

#define MAX_U64 (0xFFFFFFFFFFFFFFFF)
#define MAX_U32 (0xFFFFFFFF)

static
__device__
void
interval_compact
(
 gpu_patch_buffer_t *patch_buffer,
 gpu_patch_buffer_t *read_buffer,
 gpu_patch_buffer_t *write_buffer
)
{
  auto warp_index = blockDim.x / GPU_PATCH_WARP_SIZE * blockIdx.x + threadIdx.x / GPU_PATCH_WARP_SIZE;
  auto num_warps = blockDim.x / GPU_PATCH_WARP_SIZE;
  auto laneid = get_laneid();
  gpu_patch_record_address_t *records = (gpu_patch_record_address_t *)patch_buffer->records;
  gpu_patch_analysis_address_t *read_records = (gpu_patch_analysis_address_t *)read_buffer->records;
  gpu_patch_analysis_address_t *write_records = (gpu_patch_analysis_address_t *)write_buffer->records;

  PRINT("gpu analysis->full: %u, analysis: %u, head_index: %u, tail_index: %u, size: %u, num_threads: %u",
    patch_buffer->full, patch_buffer->analysis, patch_buffer->head_index, patch_buffer->tail_index,
    patch_buffer->size, patch_buffer->num_threads)

  for (auto iter = warp_index; iter < patch_buffer->head_index; iter += num_warps) {
    gpu_patch_record_address_t *record = records + iter;
    uint64_t address_start = record->address[laneid];
    if (((0x1u << laneid) & record->active) == 0) {
      // Those address_start does not matter
      address_start = 0;
    }

    // Sort addresses and check if they are contiguous
    address_start = warp_sort(address_start, laneid);

    // First none zero
    uint32_t b = ballot((int32_t)(address_start != 0));
    uint32_t first_laneid = __ffs(b) - 1;
    uint64_t interval_start = 0;
    interval_start = shfl_up(address_start, 1);

    PRINT_ALL("gpu_analysis <%d, %d>->active: %x, interval_start: %p, address_start: %p\n",
      blockIdx.x, threadIdx.x, record->active, interval_start, address_start);

    int32_t interval_start_point = 0;
    if (first_laneid == laneid || (address_start != 0 && (interval_start + record->size < address_start))) {
      interval_start_point = 1;
    }

    // In the worst case, a for loop takes 31 * 3 steps (shift + compare + loop) to find 
    // the right end. The following procedure find the end with ~10 instructions.
    // Find the end position
    // 00100010b
    // 76543210
    //       x
    // laneid = 1
    b = ballot(interval_start_point);

    PRINT_ALL("gpu_analysis <%d, %d>->ballot: %x, interval_start_point: %d, address_start: %p\n",
      blockIdx.x, threadIdx.x, b, interval_start_point, address_start);

    // 00100010b
    // b_rev
    // 01000100b
    // 76543210
    //  x
    // laneid_rev = 8 - 1 - 1 = 6
    uint32_t b_rev = brev(b);
    uint32_t laneid_rev = GPU_PATCH_WARP_SIZE - laneid - 1; 
    uint32_t laneid_rev_mask = (1 << laneid_rev) - 1;

    PRINT_ALL("gpu_analysis <%d, %d>->b_rev: %x, laneid_rev: %x, laneid_rev_mask: %x\n",
      blockIdx.x, threadIdx.x, b_rev, laneid_rev, laneid_rev_mask);

    // 00000100b
    // 76543210
    //      x
    // p_rev = 2
    // p = 8 - 2 - 1 = 5
    uint32_t p = bfind(laneid_rev_mask & b_rev);
    if (p != MAX_U32) {
      // Get the end of the interval
      // max(p) = 30
      p = GPU_PATCH_WARP_SIZE - p - 1 - 1;
    } else {
      // Get last
      p = GPU_PATCH_WARP_SIZE - 1;
    }
    uint64_t address_end = address_start + record->size;
    address_end = shfl(address_end, p);
    
    PRINT_ALL("gpu_analysis <%d, %d>->p: %d, address_start: %p, address_end: %p\n",
      blockIdx.x, threadIdx.x, p, address_start, address_end);

    if (interval_start_point == 1) {
      gpu_patch_analysis_address_t *address_record = NULL;

      if (record->flags & GPU_PATCH_READ) {
        address_record = read_records + gpu_queue_get(read_buffer); 
        address_record->start = address_start;
        address_record->end = address_end;

        PRINT_ALL("gpu_analysis <%d, %d>->push address_start: %p, address_end: %p\n",
          blockIdx.x, threadIdx.x, address_start, address_end);
        gpu_queue_push(read_buffer);
      } 
      
      if (record->flags & GPU_PATCH_WRITE) {
        address_record = write_records + gpu_queue_get(write_buffer); 
        address_record->start = address_start;
        address_record->end = address_end;

        PRINT_ALL("gpu_analysis <%d, %d>->push address_start: %p, address_end: %p\n",
          blockIdx.x, threadIdx.x, address_start, address_end);
        gpu_queue_push(write_buffer);
      } 
    }
  }
}


template<int THREADS, int ITEMS>
static
__device__
int
interval_merge_impl
(
 uint64_t *d_in,
 uint64_t *d_out,
 uint32_t valid_items
)
{
  // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef cub::BlockLoad<uint64_t, THREADS, ITEMS, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
  // Specialize BlockStore type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef cub::BlockStore<uint64_t, THREADS, ITEMS, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;
  // Specialize BlockRadixSort type for our thread block
  typedef cub::BlockRadixSort<uint64_t, THREADS, ITEMS, int> BlockRadixSortT;
  // Specialize BlockScan type for our thread block
  typedef cub::BlockScan<int, THREADS> BlockScanT;
  // Specialize BlockDiscontinuity for a 1D block of 128 threads on type int
  typedef cub::BlockDiscontinuity<int, THREADS> BlockDiscontinuity;
  // Shared memory
  __shared__ union TempStorage
  {
    typename BlockLoadT::TempStorage         load;
    typename BlockStoreT::TempStorage        store;
    typename BlockRadixSortT::TempStorage    sort;
    typename BlockScanT::TempStorage         scan;
    typename BlockDiscontinuity::TempStorage disc;
  } temp_storage;

  // Per-thread tile items
  uint64_t items[ITEMS];
  int interval_start_point[ITEMS];
  int interval_end_point[ITEMS];
  int interval_start_index[ITEMS];
  int interval_end_index[ITEMS];

  // Load items into a blocked arrangement
  BlockLoadT(temp_storage.load).Load(d_in, items, valid_items, MAX_U64);
  __syncthreads();

  for (uint32_t i = 0; i < ITEMS / 2; ++i) {
    if (items[i * 2] != MAX_U64) {
      items[i * 2] = items[i * 2] << 1;
    }
    if (items[i * 2 + 1] != MAX_U64) {
      items[i * 2 + 1] = (items[i * 2 + 1] << 1) + 1;
    }
  }

  for (uint32_t i = 0; i < ITEMS / 2; ++i) {
    if (items[i * 2] != MAX_U64) {
      interval_start_point[i * 2] = 1;
    } else {
      interval_start_point[i * 2] = 0;
    }
    if (items[i * 2 + 1] != MAX_U64) {
      interval_start_point[i * 2 + 1] = -1;
    } else {
      interval_start_point[i * 2 + 1] = 0;
    }
    interval_end_point[i * 2] = 0;
    interval_end_point[i * 2 + 1] = 0;
    interval_start_index[i * 2] = 0;
    interval_start_index[i * 2 + 1] = 0;
    interval_end_index[i * 2] = 0;
    interval_end_index[i * 2 + 1] = 0;
  }

  // Sort keys
  BlockRadixSortT(temp_storage.sort).Sort(items, interval_start_point);
  __syncthreads();

  // Get end marks
  BlockScanT(temp_storage.scan).InclusiveSum(interval_start_point, interval_start_point);
  __syncthreads();

  for (uint32_t i = 0; i < ITEMS; ++i) {
    if (items[i] != MAX_U64 && interval_start_point[i] == 0) {
      interval_end_point[i] = 1;
    }
  }

  // Get start marks
  // XXX(Keren): this interface has a different input and output order.
  BlockDiscontinuity(temp_storage.disc).FlagHeads(interval_start_point, interval_end_point, cub::Inequality());
  __syncthreads();

  for (uint32_t i = 0; i < ITEMS; ++i) {
    if (items[i] != MAX_U64 && interval_start_point[i] == 1 && interval_end_point[i] != 1) {
      interval_start_point[i] = 1;
    } else {
      interval_start_point[i] = 0;
    }
  }

  // Get interval start index
  int aggregate = 0;
  BlockScanT(temp_storage.scan).InclusiveSum(interval_start_point, interval_start_index, aggregate);
  __syncthreads();

  // Get interval end index
  BlockScanT(temp_storage.scan).InclusiveSum(interval_end_point, interval_end_index);
  __syncthreads();

  // Put indices in the corresponding slots
  for (uint32_t i = 0; i < ITEMS; ++i) {
    if (interval_start_point[i] == 1) {
      d_out[(interval_start_index[i] - 1) * 2] = (items[i] >> 1);
    }
    if (interval_end_point[i] == 1) {
      d_out[(interval_end_index[i] - 1) * 2 + 1] = (items[i] - 1) >> 1;
    }
  }

  return aggregate;
}


template<int THREADS, int ITEMS>
static
__device__
void
interval_merge
(
 gpu_patch_buffer_t *buffer
)
{
  uint32_t cur_index = 0;
  uint32_t items = 0;
  uint32_t tile_size = THREADS * ITEMS;
  uint64_t *records = (uint64_t *)buffer->records;
  for (; cur_index + (tile_size / 2) <= buffer->head_index; cur_index += (tile_size / 2)) {
    items += interval_merge_impl<THREADS, ITEMS>(records + cur_index * 2, records + items * 2, tile_size);
    PRINT("gpu analysis-> head_index %u, cur_index %u, tile_size %u, items %u\n", buffer->head_index, cur_index, tile_size, items);
    __syncthreads();
  }
  // Remainder
  if (cur_index < buffer->head_index) {
    items += interval_merge_impl<THREADS, ITEMS>(records + cur_index * 2, records + items * 2, ((buffer->head_index - cur_index) * 2));
    PRINT("gpu analysis-> head_index %u, cur_index %u, tile_size %u, items %u\n", buffer->head_index, cur_index, (buffer->head_index - cur_index) * 2, items);
    __syncthreads();
  }

  // Second pass
  // Fake shuffle
  if (items < buffer->head_index) {
    cur_index = 0;
    items = 0;
    for (; cur_index + (tile_size / 2) <= buffer->head_index; cur_index += (tile_size / 2)) {
      items += interval_merge_impl<THREADS, ITEMS>(records + cur_index * 2, records + items * 2, tile_size);
      PRINT("gpu analysis-> head_index %u, cur_index %u, tile_size %u, items %u\n", buffer->head_index, cur_index, tile_size, items);
      __syncthreads();
    }
    // Remainder
    if (cur_index < buffer->head_index) {
      items += interval_merge_impl<THREADS, ITEMS>(records + cur_index * 2, records + items * 2, ((buffer->head_index - cur_index) * 2));
      PRINT("gpu analysis-> head_index %u, cur_index %u, tile_size %u, items %u\n", buffer->head_index, cur_index, (buffer->head_index - cur_index) * 2, items);
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    buffer->head_index = items;
    buffer->tail_index = items;
  }
}


// TODO(Keren): multiple buffers, no need to wait
extern "C"
__launch_bounds__(GPU_PATCH_ANALYSIS_THREADS, 1)
__global__
void
gpu_analysis_interval_merge
(
 gpu_patch_buffer_t *buffer,
 gpu_patch_buffer_t *read_buffer,
 gpu_patch_buffer_t *write_buffer
)
{
  // Continue processing until CPU notifies analysis is done
  while (true) {
    // Wait until GPU notifies buffer is full. i.e., analysis can begin process.
    // Block sampling is not allowed
    while (buffer->analysis == 0 && atomic_load(&buffer->num_threads) != 0);

    if (atomic_load(&buffer->num_threads) == 0) {
      // buffer->analysis must be 0
      break;
    }

    // Compact addresses from contiguous thread accesses within each warp
    interval_compact(buffer, read_buffer, write_buffer);

    // Compact is done
    __syncthreads();

    if (threadIdx.x == 0) {
      buffer->analysis = 0;
    }

    // Merge read buffer
    if (read_buffer->head_index != 0) {
      interval_merge<GPU_PATCH_ANALYSIS_THREADS, GPU_PATCH_ANALYSIS_ITEMS>(read_buffer);

      PRINT("gpu analysis-> read buffer\n")
      PRINT_RECORDS(read_buffer)
    }

    // Merge write buffer
    if (write_buffer->head_index != 0) {
      interval_merge<GPU_PATCH_ANALYSIS_THREADS, GPU_PATCH_ANALYSIS_ITEMS>(write_buffer);

      PRINT("gpu analysis-> write buffer\n")
      PRINT_RECORDS(write_buffer)
    }

    __syncthreads();
  }

  // Last analysis
  interval_compact(buffer, read_buffer, write_buffer);

  // Compact is done
  __syncthreads();

  // Merge read buffer
  if (read_buffer->head_index != 0) {
    interval_merge<GPU_PATCH_ANALYSIS_THREADS, GPU_PATCH_ANALYSIS_ITEMS>(read_buffer);

    PRINT("gpu analysis-> read buffer\n")
    PRINT_RECORDS(read_buffer)
  }

  // Merge write buffer
  if (write_buffer->head_index != 0) {
    interval_merge<GPU_PATCH_ANALYSIS_THREADS, GPU_PATCH_ANALYSIS_ITEMS>(write_buffer);

    PRINT("gpu analysis-> write buffer\n")
    PRINT_RECORDS(write_buffer)
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    atomic_store_system(&read_buffer->num_threads, (uint32_t)0);
  }
}
