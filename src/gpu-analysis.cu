/*
 * Use C style programming in this file
 */
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
#else
#define PRINT(...)
#define PRINT_ALL(...)
#endif


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
  auto num_warps = blockDim.x / GPU_PATCH_WARP_SIZE * gridDim.x;
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
    address_start = warp_sort(address_start);
    uint32_t first_laneid = __ffs(record->active) - 1;
    uint64_t interval_start = shfl_up(address_start, 1);

    PRINT_ALL("gpu_analysis <%d, %d>->active: %x, interval_start: %p, address_start: %p\n",
      blockIdx.x, threadIdx.x, record->active, interval_start, address_start);

    int32_t interval_start_point = 0;
    if (address_start != 0 && (interval_start + record->size != address_start)) {
      interval_start_point = 1;
    }

    // In the worst case, a for loop takes 31 * 3 steps (shift + compare + loop) to find 
    // the right end. The following procedure find the end with ~10 instructions.
    // Find the end position
    // 00100010b
    // 76543210
    //       x
    // laneid = 1
    uint32_t b = ballot(interval_start_point);

    PRINT_ALL("gpu_analysis <%d, %d>->ballot: %x, interval_start_point: %d, address_start: %p\n",
      blockIdx.x, threadIdx.x, b, interval_start_point, address_start);

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
    if (p != 0xFFFFFFFF) {
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

#define ITEMS 16

#if 0
template<typename KEY>
__device__
void
interval_merge
(
 KEY *d_in,
 KEY *d_out
)
{
	enum { TILE_SIZE = THREADS * ITEMS };
	// Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
	typedef cub::BlockLoad<KEY, THREADS, ITEMS, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
  // Specialize BlockStore type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef cub::BlockStore<KEY, THREADS, ITEMS, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;
	// Specialize BlockRadixSort type for our thread block
	typedef cub::BlockRadixSort<KEY, THREADS, ITEMS, int> BlockRadixSortT;
  // Specialize BlockScan type for our thread block
  typedef cub::BlockScan<int, THREADS> BlockScanT;
	// Shared memory
	__shared__ union TempStorage
	{
		typename BlockLoadT::TempStorage        load;
    typename BlockStoreT::TempStorage       store;
		typename BlockRadixSortT::TempStorage   sort;
    typename BlockScanT::TempStorage        scan;
	} temp_storage;

	// Per-thread tile items
	KEY items[ITEMS];
  int interval_start[ITEMS];
  int interval_end[ITEMS];
  int interval_start_index[ITEMS];
  int interval_end_index[ITEMS];

	// Load items into a blocked arrangement
	BlockLoadT(temp_storage.load).Load(d_in, items);

  for (size_t i = 0; i < ITEMS / 2; ++i) {
    items[i * 2 + 1] += 1;
  }

	// Barrier for smem reuse
	__syncthreads();

  for (size_t i = 0; i < ITEMS / 2; ++i) {
    interval_start[i * 2] = 1;
    interval_start[i * 2 + 1] = -1;
    interval_end[i * 2] = 0;
    interval_end[i * 2 + 1] = 0;
    interval_start_index[i * 2] = 0;
    interval_start_index[i * 2 + 1] = 0;
    interval_end_index[i * 2] = 0;
    interval_end_index[i * 2 + 1] = 0;
  }

	// Sort keys
	BlockRadixSortT(temp_storage.sort).Sort(items, interval_start);
  __syncthreads();

  // Get start/end marks
  BlockScanT(temp_storage.scan).InclusiveSum(interval_start, interval_start);
  __syncthreads();

  for (size_t i = 0; i < ITEMS; ++i) {
    if (interval_start[i] == 1) {
      // do nothing
    } else if (interval_start[i] == 0) {
      interval_end[i] = 1;
    } else {
      interval_start[i] = 0;
    }
  }
  // Get interval start index
  BlockScanT(temp_storage.scan).InclusiveSum(interval_start, interval_start_index);
  __syncthreads();

  // Get interval end index
  BlockScanT(temp_storage.scan).InclusiveSum(interval_end, interval_end_index);
  __syncthreads();

  // Put indices in the corresponding slots
  for (size_t i = 0; i < ITEMS; ++i) {
    if (interval_start[i] == 1) {
      d_out[(interval_start_index[i] - 1) * 2] = items[i];
    }
    if (interval_end[i] == 1) {
      d_out[(interval_end_index[i] - 1) * 2 + 1] = items[i] - 1;
    }
  }
}
#endif

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
  while (read_buffer->analysis == 1 || write_buffer->analysis == 1) {
		// Wait until GPU notifies buffer is full. i.e., analysis can begin process.
    // Block sampling is not allowed
    while (buffer->analysis == 0 && atomic_load(&buffer->num_threads) != 0); 

    // Compact addresses from contiguous thread accesses within each warp
    interval_compact(buffer, read_buffer, write_buffer);

		// TODO(Keren): ensure compact is done by all blocks
    // Compact is done
    __syncthreads();
    __threadfence_system();
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			buffer->analysis = 0;
		}
    __threadfence_system();
    __syncthreads();

    //// Merge read buffer
    //interval_merge<uint64_t, GPU_PATCH_THREADS, GPU_PATCH_ITEMS>(read_buffer);
    //// Merge write buffer
    //interval_merge<uint64_t, GPU_PATCH_THREADS, GPU_PATCH_ITEMS>(write_buffer);
  }
}
