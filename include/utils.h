#ifndef HPCTOOLKIT_GPU_PATCH_UTILITIES_H
#define HPCTOOLKIT_GPU_PATCH_UTILITIES_H

#include <stdint.h>

/*
 * Utility functions
 */
__device__ __forceinline__
uint32_t
get_flat_block_id
(
)
{
  return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
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
uint64_t
get_unique_thread_id
(
)
{
  return get_flat_block_id() * blockDim.x * blockDim.y * blockDim.z + get_flat_thread_id();
}


__device__ __forceinline__
uint32_t
get_laneid
(
)
{
  uint32_t laneid = 0;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid));
  return laneid;
}


__device__ __forceinline__
bool
sample_callback
(
 uint32_t frequency,
 uint32_t offset
)
{
  if (frequency != 0) {
    // 1  : Sample all blocks
    // >1 : Sample a portion of blocks
    return get_flat_block_id() % frequency == offset;
  }
  // Skip all blocks
  return false;
}


__device__ __forceinline__
bool
is_locked
(
 uint32_t *lock,
 uint32_t id
)
{
  uint32_t old = *lock;
  // Read the newest value
  __threadfence();
  return old == id;
}


__device__ __forceinline__
void
read_shared_memory
(
 uint32_t size,
 uint32_t ptr,
 uint8_t *buf
)
{
  for (uint32_t i = 0; i < size; ++i) {
    uint32_t ret = 0;
    asm volatile("ld.shared.b8 %0,[%1];" : "=r"(ret) : "r"(ptr + i) : "memory");
    buf[i] = ret;
  }
}


__device__ __forceinline__
void
read_global_memory
(
 uint32_t size,
 uint64_t ptr,
 uint8_t *buf
)
{
  for (uint32_t i = 0; i < size; ++i) {
    uint32_t ret = 0;
    asm volatile("ld.b8 %0,[%1];" : "=r"(ret) : "l"(ptr + i) : "memory");
    buf[i] = ret;
  }
}


__device__ __forceinline__
void
read_local_memory
(
 uint32_t size,
 uint32_t ptr,
 uint8_t *buf
)
{
  for (uint32_t i = 0; i < size; ++i) {
    uint32_t ret = 0;
    asm volatile("ld.local.b8 %0,[%1];" : "=r"(ret) : "r"(ptr + i) : "memory");
    buf[i] = ret;
  }
}


template<class T>
__device__ __forceinline__
T
shfl
(
 T v,
 uint32_t srcline,
 uint32_t mask = 0xFFFFFFFF
)
{
  T ret;
#if (__CUDA_ARCH__ >= 300)
#if (__CUDACC_VER_MAJOR__ >= 9)
    ret = __shfl_sync(mask, v, srcline);
#else
    ret = __shfl(v, srcline);
#endif
#endif
  return ret;
}


__device__ __forceinline__
uint32_t
ballot
(
 int32_t predicate,
 uint32_t mask = 0xFFFFFFFF
)
{
  uint32_t ret;
#if (__CUDA_ARCH__ >= 300)
#if (__CUDACC_VER_MAJOR__ >= 9)
  ret = __ballot_sync(mask, predicate);
#else
  ret = __ballot(predicate);
#endif
#endif
  return ret;
}

#endif
