#ifndef HPCTOOLKIT_GPU_MEMORY_PATCH_MEMORY_H
#define HPCTOOLKIT_GPU_MEMORY_PATCH_MEMORY_H

#include <cstdint>
#include <vector_types.h>

#define MAX_ACCESS_SIZE (16)
#define THREAD_HASH_SIZE (128 * 1024 - 1)

typedef struct sanitizer_memory_buffer {
  uint64_t pc;
  uint64_t address;
  uint32_t size;
  uint32_t flags;
  uint8_t value[MAX_ACCESS_SIZE];  // STS.128->16 bytes
  dim3 thread_ids;
  dim3 block_ids;
} sanitizer_memory_buffer_t;


typedef struct sanitizer_buffer {
  uint32_t cur_index;
  uint32_t max_index;
  uint32_t *thread_hash_locks;  // max thread id > 2^31
  uint32_t block_sampling_frequency;
  void **prev_memory_buffer;
  void *buffers;
} sanitizer_buffer_t;


#endif
