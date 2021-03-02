#ifndef HPCTOOLKIT_GPU_PATCH_GPU_QUEUE_H
#define HPCTOOLKIT_GPU_PATCH_GPU_QUEUE_H

#include <stdint.h>

#include "gpu-patch.h"

/*
 * Get a gpu record
 */
extern "C" __device__ uint32_t gpu_queue_get(gpu_patch_buffer_t *buffer, uint32_t analysis = 0) {
  uint32_t size = buffer->size;
  uint32_t tail_index = 0;
  while (tail_index == 0) {
    tail_index = atomicAdd((uint32_t *)&buffer->tail_index, 1) + 1;
    // Write on tail_index - 1
    if (tail_index - 1 >= size) {
      // First warp that found buffer is full
      if (tail_index - 1 == size) {
        // Wait for previous warps finish writing
        while (buffer->head_index < size);
        if (analysis == 1) {
          // Sync with GPU
          __threadfence();
          buffer->analysis = 1;
          __threadfence();
          while (buffer->analysis == 1)
            ;
        } else {
          // Sync with CPU
          __threadfence_system();
          buffer->full = 1;
          __threadfence_system();
          while (buffer->full == 1);
        }
        __threadfence();
        buffer->head_index = 0;
        __threadfence();
        buffer->tail_index = 0;
      } else {
        // Other waps
        while (buffer->tail_index >= size);
      }
      tail_index = 0;
    }
  }

  return tail_index - 1;
}

/*
 * Finish writing gpu records
 */
extern "C" __device__ void gpu_queue_push(gpu_patch_buffer_t *buffer) {
  // Make sure records are visible
  atomicAdd((uint32_t *)&buffer->head_index, 1);
}

#endif
