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
  uint32_t block_id = get_flat_block_id();
  uint32_t thread_id = get_flat_thread_id();
  uint32_t block_hash_index = block_id % BLOCK_HASH_SIZE;
  uint32_t thread_hash_index = block_hash_index * MAX_BLOCK_THREADS + thread_id;

  // Get prev ptr and size
  sanitizer_memory_buffer_t *prev_memory_buffer = buffer->prev_memory_buffer[thread_hash_index];

  if (prev_memory_buffer != NULL) {
    char *prev_ptr = (char *)(void *)prev_memory_buffer->address;
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
  uint32_t block_id = get_flat_block_id();
  uint32_t thread_id = get_flat_thread_id();
  uint32_t block_hash_index = block_id % BLOCK_HASH_SIZE;
  uint32_t thread_hash_index = block_hash_index * MAX_BLOCK_THREADS + thread_id;

  // Spin lock wait
  // TODO(keren): backoff?
  acquire(&buffer->block_hash_locks[block_hash_index]);

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

  // Update prev memory accesses
  update_prev_memory_buffer(buffer, NULL);

  uint32_t block_id = get_flat_block_id();
  uint32_t block_hash_index = block_id % BLOCK_HASH_SIZE;

  release(&buffer->block_hash_locks[block_hash_index]);

  return SANITIZER_PATCH_SUCCESS;
}


