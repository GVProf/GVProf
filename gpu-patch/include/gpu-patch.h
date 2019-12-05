#ifndef HPCTOOLKIT_GPU_PATCH_GPU_PATCH_H
#define HPCTOOLKIT_GPU_PATCH_GPU_PATCH_H

#include <stdint.h>

#define MAX_ACCESS_SIZE (16)
#define WARP_SIZE (32)

typedef struct gpu_patch_record {
  uint64_t pc;
  uint32_t size;
  uint32_t active;
  uint32_t flat_thread_id;
  uint32_t flat_block_id;
  uint64_t address[WARP_SIZE];
  uint8_t value[WARP_SIZE][MAX_ACCESS_SIZE];  // STS.128->16 bytes
#ifdef DEBUG
  uint32_t flag[WARP_SIZE];
#endif
} gpu_patch_record_t;


typedef struct gpu_patch_buffer {
  uint32_t head_index;
  uint32_t tail_index;
  uint32_t size;
  uint32_t nblocks;  // If nblocks == 0, the kernel is finished
  bool full;
  uint32_t block_sampling_frequency;
  void *gpu_patch_records;
} gpu_patch_buffer_t;

#endif
