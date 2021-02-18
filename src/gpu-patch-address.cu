/*
 * Use C style programming in this file
 */
#include "gpu-patch.h"
#include "gpu-queue.h"
#include "utils.h"

#include <sanitizer_patching.h>

/*
 * Monitor each shared and global memory access.
 */
static 
__device__ __forceinline__
SanitizerPatchResult
memory_access_callback
(
 void *user_data,
 uint64_t pc,
 void *address,
 uint32_t size,
 uint32_t flags,
 const void *new_value
) 
{
  gpu_patch_buffer_t *buffer = (gpu_patch_buffer_t *)user_data;

  // 1. Init values
  uint32_t active_mask = __activemask();
  uint32_t laneid = get_laneid();
  uint32_t first_laneid = __ffs(active_mask) - 1;

  gpu_patch_record_t *record = NULL;
  if (laneid == first_laneid) {
    // 3. Get a record
    gpu_patch_record_t *records = (gpu_patch_record_t *)buffer->records;
    record = records + gpu_queue_get(buffer, buffer->flags & GPU_PATCH_ANALYSIS); 

    // 4. Assign basic values
    record->flags = flags;
    record->active = active_mask;
  }

  __syncwarp(active_mask);

  uint64_t r = (uint64_t)record;
  record = (gpu_patch_record_t *)shfl(r, first_laneid, active_mask);

  if (record != NULL) {
    record->address[laneid] = (uint64_t)address;
  }

  __syncwarp(active_mask);

  if (laneid == first_laneid) {
    // 5. Push a record
    gpu_queue_push(buffer);
  }

  return SANITIZER_PATCH_SUCCESS;
}


extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_global_memory_access_callback
(
 void *user_data,
 uint64_t pc,
 void *address,
 uint32_t size,
 uint32_t flags,
 const void *new_value
) 
{
  return memory_access_callback(user_data, pc, address, size, flags, new_value);
}


extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_shared_memory_access_callback
(
 void *user_data,
 uint64_t pc,
 void *address,
 uint32_t size,
 uint32_t flags,
 const void *new_value
) 
{
  return memory_access_callback(user_data, pc, address, size, flags | GPU_PATCH_SHARED, new_value);
}


extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_local_memory_access_callback
(
 void *user_data,
 uint64_t pc,
 void *address,
 uint32_t size,
 uint32_t flags,
 const void *new_value
) 
{
  return memory_access_callback(user_data, pc, address, size, flags | GPU_PATCH_LOCAL, new_value);
}


/*
 * Lock the corresponding hash entry for a block
 */
extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_block_exit_callback
(
 void *user_data,
 uint64_t pc
)
{
  gpu_patch_buffer_t* buffer = (gpu_patch_buffer_t *)user_data;

  if (!sample_callback(buffer->block_sampling_frequency, buffer->block_sampling_offset)) {
    return SANITIZER_PATCH_SUCCESS;
  }

  uint32_t active_mask = __activemask();
  uint32_t laneid = get_laneid();
  uint32_t first_laneid = __ffs(active_mask) - 1;
  int32_t pop_count = __popc(active_mask);

  if (laneid == first_laneid) {
    // Finish a bunch of threads
    atomicAdd(&buffer->num_threads, -pop_count);
  }

  return SANITIZER_PATCH_SUCCESS;
}

