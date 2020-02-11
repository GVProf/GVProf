#ifndef HPCTOOLKIT_GPU_PATCH_GPU_PATCH_H
#define HPCTOOLKIT_GPU_PATCH_GPU_PATCH_H

#include <stdint.h>
#include <stdbool.h>

#define GPU_PATCH_MAX_ACCESS_SIZE (16)
#define GPU_PATCH_WARP_SIZE (32)
#define GPU_PATCH_BLOCK_ENTER_FLAG (0x20)
#define GPU_PATCH_BLOCK_EXIT_FLAG (0x40)

/*
 * SANITIZER_MEMORY_DEVICE_FLAG_NONE      = 0,
 * SANITIZER_MEMORY_DEVICE_FLAG_READ      = 0x1,
 * SANITIZER_MEMORY_DEVICE_FLAG_WRITE     = 0x2,
 * SANITIZER_MEMORY_DEVICE_FLAG_ATOMSYS   = 0x4,
 * SANITIZER_MEMORY_DEVICE_FLAG_LOCAL     = 0x8,
 * SANITIZER_MEMORY_DEVICE_FLAG_SHARED    = 0x10,
 */

typedef struct gpu_patch_record {
  uint64_t pc;
  uint32_t size;
  uint32_t active;
  uint32_t flat_thread_id;
  uint32_t flat_block_id;
  uint64_t address[GPU_PATCH_WARP_SIZE];
  uint8_t value[GPU_PATCH_WARP_SIZE][GPU_PATCH_MAX_ACCESS_SIZE];  // STS.128->16 bytes
  uint32_t flags;
} gpu_patch_record_t;


typedef struct gpu_patch_buffer {
  volatile uint32_t head_index;
  volatile uint32_t tail_index;
  uint32_t size;
  uint32_t num_blocks;  // If nblocks == 0, the kernel is finished
  uint32_t block_sampling_offset;
  uint32_t block_sampling_frequency;
  volatile uint32_t full;
  void *records;
} gpu_patch_buffer_t;

#endif
