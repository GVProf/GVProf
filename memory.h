#ifndef HPCTOOLKIT_GPU_MEMORY_PATCH_MEMORY_H
#define HPCTOOLKIT_GPU_MEMORY_PATCH_MEMORY_H

#include <cstdint>
#include <vector_types.h>

#define MAX_BLOCK_THREADS 1024
#define MAX_ACCESS_SIZE 128
#define BLOCK_HASH_SIZE 4095

typedef struct sanitizer_memory_buffer {
  uint64_t pc;
  uint64_t address;
  uint32_t size;
  uint32_t flags;
  char value[MAX_ACCESS_SIZE];  // STS.128->16 bytes
  dim3 thread_ids;
  dim3 block_ids;
} sanitizer_memory_buffer_t;


typedef struct sanitizer_buffer {
  uint32_t cur_index;
  uint32_t max_index;
  int block_hash_locks[BLOCK_HASH_SIZE];
  sanitizer_memory_buffer_t *prev_memory_buffer[BLOCK_HASH_SIZE * MAX_BLOCK_THREADS];
  void *buffers;
} sanitizer_buffer_t;


#endif
