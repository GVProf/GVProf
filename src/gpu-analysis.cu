/*
 * Use C style programming in this file
 */
#include "gpu-patch.h"
#include "gpu-queue.h"
#include "utils.h"

#include <cub/cub.cuh>

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

  for (auto iter = warp_index; iter < patch_buffer->head_index; iter += num_warps) {
    gpu_patch_record_address_t *record = records + iter;
    uint64_t address_start = record->address[threadIdx.x];
    if ((laneid & record->active) == 0) {
      // Those address_start does not matter
      address_start = 0;
    }

    address_start = warp_sort(address_start);
    uint32_t first_laneid = __ffs(record->active) - 1;
    uint32_t interval_start = shfl_up(record->active, address_start, 1);
    if (address_start != 0 && (interval_start + record->size != address_start || laneid == first_laneid)) {
      interval_start = 1;
    } else {
      interval_start = 0;
    }

    // find the end position
    uint32_t b = ballot(record->active, interval_start);
    uint32_t p = fns(b, laneid + 1);
    if (p == 0xFFFFFFFF) {
      p = GPU_PATCH_WARP_SIZE - 1;
    }
    uint64_t address_end = shfl(address_start + record->size, p);

    if (interval_start == 1) {
      gpu_patch_analysis_address_t *address_record = NULL;

      if (record->flags & GPU_PATCH_READ) {
        address_record = read_records + gpu_queue_get(read_buffer); 
      } else {
        address_record = write_records + gpu_queue_get(write_buffer); 
      } 
      address_record->start = address_start;
      address_record->end = address_end;
      if (record->flags & GPU_PATCH_READ) {
        gpu_queue_push(read_buffer);
      } else {
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
  // Continue processing until analysis is 0
  while (read_buffer->analysis == 1) {
    while (buffer->analysis == 0 && read_buffer->analysis == 1);
    // Compact address
    interval_compact(buffer, read_buffer, write_buffer);

    // Compact is done
    __syncthreads();
    __threadfence_system();
    buffer->analysis = 0;
    __threadfence_system();
    __syncthreads();

    //// Merge read buffer
    //interval_merge<uint64_t, GPU_PATCH_THREADS, GPU_PATCH_ITEMS>(read_buffer);
    //// Merge write buffer
    //interval_merge<uint64_t, GPU_PATCH_THREADS, GPU_PATCH_ITEMS>(write_buffer);
  }
}
