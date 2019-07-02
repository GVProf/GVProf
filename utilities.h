#ifndef HPCTOOLKIT_GPU_MEMORY_PATCH_UTILITIES_H
#define HPCTOOLKIT_GPU_MEMORY_PATCH_UTILITIES_H

/*
 * Utility functions
 */
__device__ __forceinline__
uint32_t
get_flat_block_id
(
)
{
  return blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
}


__device__ __forceinline__
uint32_t
get_flat_thread_id
(
)
{
  return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}


__device__ __forceinline__
void
acquire
(
 volatile int *lock
)
{
  while (atomicCAS((int *)lock, 0, 1) != 0);
}


__device__ __forceinline__
void 
release
(
 volatile int *lock
)
{
  *lock = 0;
  __threadfence();
}

#endif
