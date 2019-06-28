#ifndef HPCTOOLKIT_GPU_MEMORY_PATCH
#define HPCTOOLKIT_GPU_MEMORY_PATCH

#include <cstdint>
#include <vector_types.h>

typedef struct sanitizer_memory_buffer {
  uint64_t pc;
  uint64_t address;
  uint32_t size;
  uint32_t flags;
  dim3 thread_ids;
  dim3 block_ids;
} sanitizer_memory_buffer_t;


typedef struct sanitizer_buffer {
  uint32_t cur_index;
  uint32_t max_index;
  void *buffers;
} sanitizer_buffer_t;


#endif
