#include "memory.h"

#include <sanitizer_patching.h>

__device__ __forceinline__
uint32_t
get_flat_block_id()
{
  return blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
}


__device__ __forceinline__
uint32_t
get_flat_thread_id()
{
  return threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

// Use C style programming
extern "C" __device__ __noinline__
SanitizerPatchResult
sanitizer_memory_callback
(
 void* userdata,
 uint64_t pc,
 void* ptr,
 uint32_t size,
 uint32_t flags
) 
{
  sanitizer_buffer_t* buffer = (sanitizer_buffer_t *)userdata;

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

  // Compute thread id
  uint32_t block_id = get_flat_block_id();
  uint32_t thread_id = get_flat_thread_id();
  uint32_t block_hash_index = block_id % BLOCK_HASH_SIZE;
  uint32_t thread_hash_index = block_hash_index * MAX_BLOCK_THREADS + thread_id;

  // Get prev ptr and index
  void *prev_ptr = buffer->prev_ptr[thread_hash_index];
  uint32_t prev_index = buffer->prev_index[thread_hash_index];
  uint32_t prev_size = buffer->prev_size[thread_hash_index];

  if (prev_ptr != NULL) {
    sanitizer_memory_buffer_t *prev_memory_buffer = &(memory_buffers[prev_index]);

    char *byte_ptr = (char *)prev_ptr;
    for (size_t i = 0; i < prev_size; ++i) {
      prev_memory_buffer->value[i] = byte_ptr[i];
    }
  }

  // Update prev ptr and index
  buffer->prev_ptr[thread_hash_index] = ptr;
  buffer->prev_index[thread_hash_index] = cur_index;

  return SANITIZER_PATCH_SUCCESS;
}


// TODO(Keren) Add block enter or block exit instrumentation

// TODO(Keren) Add sync instrumentation
