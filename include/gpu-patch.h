#ifndef HPCTOOLKIT_GPU_PATCH_GPU_PATCH_H
#define HPCTOOLKIT_GPU_PATCH_GPU_PATCH_H

#include <stdbool.h>
#include <stdint.h>

#define GPU_PATCH_MAX_ACCESS_SIZE (16)
#define GPU_PATCH_WARP_SIZE (32)
#define GPU_PATCH_ANALYSIS_THREADS (1024)
#define GPU_PATCH_ANALYSIS_ITEMS (4)

enum GPUPatchFlags {
  GPU_PATCH_NONE = 0,
  GPU_PATCH_READ = 0x1,
  GPU_PATCH_WRITE = 0x2,
  GPU_PATCH_ATOMSYS = 0x4,
  GPU_PATCH_LOCAL = 0x8,
  GPU_PATCH_SHARED = 0x10,
  GPU_PATCH_BLOCK_ENTER_FLAG = 0x20,
  GPU_PATCH_BLOCK_EXIT_FLAG = 0x40,
  GPU_PATCH_ANALYSIS = 0x80
};

enum GPUPatchType {
  GPU_PATCH_TYPE_DEFAULT = 0,
  GPU_PATCH_TYPE_ADDRESS_PATCH = 1,
  GPU_PATCH_TYPE_ADDRESS_ANALYSIS = 2,
  GPU_PATCH_TYPE_COUNT = 3
};

// Complete record
typedef struct gpu_patch_record {
  uint64_t pc;
  uint32_t size;
  uint32_t active;
  uint32_t flat_thread_id;
  uint32_t flat_block_id;
  uint32_t flags;
  uint64_t address[GPU_PATCH_WARP_SIZE];
  uint8_t value[GPU_PATCH_WARP_SIZE][GPU_PATCH_MAX_ACCESS_SIZE];  // STS.128->16 bytes
} gpu_patch_record_t;

// Address only
typedef struct gpu_patch_record_address {
  uint32_t flags;
  uint32_t active;
  uint32_t size;
  uint64_t address[GPU_PATCH_WARP_SIZE];
} gpu_patch_record_address_t;

// Address only, gpu analysis
typedef struct gpu_patch_analysis_address {
  uint64_t start;
  uint64_t end;
} gpu_patch_analysis_address_t;

typedef struct gpu_patch_buffer {
  volatile uint32_t full;
  volatile uint32_t analysis;
  volatile uint32_t head_index;
  volatile uint32_t tail_index;
  uint32_t size;
  uint32_t num_threads;  // If num_threads == 0, the kernel is finished
  uint32_t block_sampling_offset;
  uint32_t block_sampling_frequency;
  uint32_t type;
  uint32_t flags;  // read or write or both
  void *records;
} gpu_patch_buffer_t;

#endif
