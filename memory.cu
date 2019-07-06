/*
 * Use C style programming in this file
 */
#include "memory.h"
#include "utilities.h"

#include <sanitizer_patching.h>

/*
 * Real update
 */
__device__ __forceinline__
void
update_prev_memory_buffer
(
 sanitizer_buffer_t *buffer,
 sanitizer_memory_buffer_t *cur_memory_buffer
)
{
  // Compute thread id
  size_t unique_thread_id = get_unique_thread_id();
  uint32_t thread_hash_index = unique_thread_id % THREAD_HASH_SIZE;

  // Get prev ptr and size
  sanitizer_memory_buffer_t *prev_memory_buffer = (sanitizer_memory_buffer_t *)
    (buffer->prev_memory_buffer[thread_hash_index]);

  if (prev_memory_buffer != NULL) {
    uint8_t *prev_ptr = (uint8_t *)(void *)prev_memory_buffer->address;
    uint32_t prev_size = prev_memory_buffer->size;
    for (size_t i = 0; i < prev_size; ++i) {
      prev_memory_buffer->value[i] = prev_ptr[i];
    }
  }

  // Update
  buffer->prev_memory_buffer[thread_hash_index] = cur_memory_buffer;
}


/*
 * Monitor each shared and global memory access.
 * Each time record the previous memory access to a thread private storage.
 */
extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_memory_access_callback
(
 void* user_data,
 uint64_t pc,
 void* ptr,
 uint32_t size,
 uint32_t flags
) 
{
  sanitizer_buffer_t* buffer = (sanitizer_buffer_t *)user_data;

  if (skip_callback(buffer->block_sampling_frequency)) {
    return SANITIZER_PATCH_SUCCESS;
  }

  uint32_t cur_index = atomicAdd(&(buffer->cur_index), 1);

  if (cur_index >= buffer->max_index)
    return SANITIZER_PATCH_SUCCESS;

  // Assign basic values
  sanitizer_memory_buffer_t *memory_buffers = (sanitizer_memory_buffer_t *)buffer->buffers;
  sanitizer_memory_buffer_t *cur_memory_buffer = &(memory_buffers[cur_index]);
  cur_memory_buffer->pc = pc;
  cur_memory_buffer->address = (uint64_t)ptr;
  cur_memory_buffer->size = size;
  cur_memory_buffer->flags = flags;
  cur_memory_buffer->thread_ids = threadIdx;
  cur_memory_buffer->block_ids = blockIdx;

  // Get prev ptr and index
  update_prev_memory_buffer(buffer, cur_memory_buffer);

  return SANITIZER_PATCH_SUCCESS;
}


/*
 * Prevent data race on reading the previous memory accesses across __syncthreads and __threadfence.
 * Each time record the previous memory access to a thread private storage.
 */
extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_barrier_callback
(
 void *user_data,
 uint64_t pc,
 uint32_t bar_index
)
{
  sanitizer_buffer_t* buffer = (sanitizer_buffer_t *)user_data;

  if (skip_callback(buffer->block_sampling_frequency)) {
    return SANITIZER_PATCH_SUCCESS;
  }

  // Get prev ptr and index
  update_prev_memory_buffer(buffer, NULL);

  return SANITIZER_PATCH_SUCCESS;
}


/*
 * Lock the corresponding hash entry for a block
 */
extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_block_enter_callback
(
 void *user_data
)
{
  sanitizer_buffer_t* buffer = (sanitizer_buffer_t *)user_data;

  if (skip_callback(buffer->block_sampling_frequency)) {
    return SANITIZER_PATCH_SUCCESS;
  }

  size_t unique_thread_id = get_unique_thread_id();
  uint32_t thread_hash_index = unique_thread_id % THREAD_HASH_SIZE;

  // Spin lock wait
  // TODO(keren): backoff?
  acquire(&buffer->thread_hash_locks[thread_hash_index], unique_thread_id);

  // Clear previous hash entries
  buffer->prev_memory_buffer[thread_hash_index] = NULL;

  return SANITIZER_PATCH_SUCCESS;
}


/*
 * Release the corresponding hash entry and update the previous memory access
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
  sanitizer_buffer_t* buffer = (sanitizer_buffer_t *)user_data;

  if (skip_callback(buffer->block_sampling_frequency)) {
    return SANITIZER_PATCH_SUCCESS;
  }

  // Update prev memory accesses
  update_prev_memory_buffer(buffer, NULL);

  size_t unique_thread_id = get_unique_thread_id();
  uint32_t thread_hash_index = unique_thread_id % THREAD_HASH_SIZE;

  release(&buffer->thread_hash_locks[thread_hash_index]);

  return SANITIZER_PATCH_SUCCESS;
}


