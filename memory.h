#ifndef HPCTOOLKIT_GPU_MEMORY_PATCH
#define HPCTOOLKIT_GPU_MEMORY_PATCH

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
  bool block_hash_locks[BLOCK_HASH_SIZE];
  void *prev_ptr[BLOCK_HASH_SIZE * MAX_BLOCK_THREADS];
  uint32_t prev_index[BLOCK_HASH_SIZE * MAX_BLOCK_THREADS];
  uint32_t prev_size[BLOCK_HASH_SIZE * MAX_BLOCK_THREADS];
  void *buffers;
} sanitizer_buffer_t;


#endif
