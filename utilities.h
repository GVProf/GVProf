#ifndef HPCTOOLKIT_GPU_MEMORY_PATCH_UTILITIES_H
#define HPCTOOLKIT_GPU_MEMORY_PATCH_UTILITIES_H

#include <stdint.h>

/*
 * Utility functions
 */
__device__ __forceinline__
size_t
get_flat_block_id
(
)
{
  return blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
}


__device__ __forceinline__
size_t
get_flat_thread_id
(
)
{
  return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}


__device__ __forceinline__
size_t
get_unique_thread_id
(
)
{
  return get_flat_block_id() * blockDim.x * blockDim.y * blockDim.z + get_flat_thread_id();
}


__device__ __forceinline__
bool
skip_callback
(
 uint32_t frequency
)
{
  if (frequency != 0) {
    return get_flat_block_id() % frequency != 0;
  }
  // not sampling
  return false;
}


__device__ __forceinline__
void
acquire
(
 uint32_t *lock,
 uint32_t id
)
{
  uint32_t old = *lock;
  // Read newest value
  __threadfence();
  if (old != id) {
    while (atomicCAS(lock, 0, id) != 0);
  }
}


__device__ __forceinline__
void 
release
(
 uint32_t *lock
)
{
  atomicExch(lock, 0);
  // Visiable to all the threads
  __threadfence();
}

#endif
