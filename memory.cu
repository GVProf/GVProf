/*
 * Use C style programming in this file
 */
#include "memory.h"
#include "utilities.h"

#include <sanitizer_patching.h>


/*
 * Monitor each shared and global memory access.
 */
extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_memory_access_callback
(
 void *user_data,
 uint64_t pc,
 void *address,
 uint32_t size,
 uint32_t flags,
 const void *new_value
) 
{
  // 1. Sample limitation
  sanitizer_buffer_t* buffer = (sanitizer_buffer_t *)user_data;

  if (!sample_callback(buffer->block_sampling_frequency)) {
    return SANITIZER_PATCH_SUCCESS;
  }

  // 2. Lock limitation
  size_t unique_thread_id = get_unique_thread_id();
  uint32_t thread_hash_index = unique_thread_id % THREAD_HASH_SIZE;
  uint32_t *lock = &buffer->thread_hash_locks[thread_hash_index];

  if (!is_locked(lock, unique_thread_id + 1)) {
    return SANITIZER_PATCH_SUCCESS;
  }

  // 3. Buffer size limitation
  uint32_t cur_index = atomicAdd(&(buffer->cur_index), 1);

  if (cur_index >= buffer->max_index)
    return SANITIZER_PATCH_SUCCESS;

  // Assign basic values
  sanitizer_memory_buffer_t *memory_buffers = (sanitizer_memory_buffer_t *)buffer->buffers;
  sanitizer_memory_buffer_t *cur_memory_buffer = &(memory_buffers[cur_index]);
  cur_memory_buffer->pc = pc;
  cur_memory_buffer->address = (uint64_t)address;
  cur_memory_buffer->size = size;
  cur_memory_buffer->flags = flags;
  cur_memory_buffer->thread_ids = threadIdx;
  cur_memory_buffer->block_ids = blockIdx;

  if (new_value == NULL) {
    // Read operation, old value can be on local memory, shared memory, or global memory
    if (flags & SANITIZER_MEMORY_DEVICE_FLAG_SHARED) {
      read_shared_memory(size, (uint32_t)address, (uint8_t *)cur_memory_buffer->value);
    } else if (flags & SANITIZER_MEMORY_DEVICE_FLAG_LOCAL) {
      read_local_memory(size, (uint32_t)address, (uint8_t *)cur_memory_buffer->value);
    } else if (flags != SANITIZER_MEMORY_DEVICE_FLAG_FORCE_INT) {
      read_global_memory(size, (uint64_t)address, (uint8_t *)cur_memory_buffer->value);
    }
  } else {
    // Write operation, new value is on global memory
    read_global_memory(size, (uint64_t)new_value, (uint8_t *)cur_memory_buffer->value);
  }

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
 void *user_data,
 uint64_t pc
)
{
  sanitizer_buffer_t* buffer = (sanitizer_buffer_t *)user_data;

  if (!sample_callback(buffer->block_sampling_frequency)) {
    return SANITIZER_PATCH_SUCCESS;
  }

  size_t unique_thread_id = get_unique_thread_id();
  uint32_t thread_hash_index = unique_thread_id % THREAD_HASH_SIZE;
  uint32_t *lock = &buffer->thread_hash_locks[thread_hash_index];

  // Spin lock wait
  // 0: not locked
  // 1-N: locked by a thread
  // TODO(keren): backoff?
  try_acquire(lock, unique_thread_id + 1);

  return SANITIZER_PATCH_SUCCESS;
}


/*
 * Release the corresponding hash entry
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

  if (!sample_callback(buffer->block_sampling_frequency)) {
    return SANITIZER_PATCH_SUCCESS;
  }

  size_t unique_thread_id = get_unique_thread_id();
  uint32_t thread_hash_index = unique_thread_id % THREAD_HASH_SIZE;
  uint32_t *lock = &buffer->thread_hash_locks[thread_hash_index];

  if (is_locked(lock, unique_thread_id + 1)) {
    release(lock);
  }

  return SANITIZER_PATCH_SUCCESS;
}
