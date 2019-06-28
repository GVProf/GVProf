#include "memory.h"

#include <sanitizer_patching.h>

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

  sanitizer_memory_buffer_t *memory_buffers = (sanitizer_memory_buffer_t *)buffer->buffers;
  sanitizer_memory_buffer_t *cur_memory_buffer = &(memory_buffers[cur_index]);
  cur_memory_buffer->pc = pc;
  cur_memory_buffer->address = (uint64_t)ptr;
  cur_memory_buffer->size = size;
  cur_memory_buffer->flags = flags;
  cur_memory_buffer->thread_ids = threadIdx;
  cur_memory_buffer->block_ids = blockIdx;

  return SANITIZER_PATCH_SUCCESS;
}
