#ifndef HPCTOOLKIT_GPU_MEMORY_PATCH_UTILITIES_H
#define HPCTOOLKIT_GPU_MEMORY_PATCH_UTILITIES_H

#include <stdint.h>

#define LOCK_NUM_TRIALS 1

/*
 * Utility functions
 */
__device__ __forceinline__
size_t
get_flat_block_id
(
)
{
  return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
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
sample_callback
(
 uint32_t frequency
)
{
  if (frequency != 0) {
    // Sample a portion of blocks
    return get_flat_block_id() % frequency == 0;
  }
  // Sample all blocks
  return true;
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
bool
try_acquire
(
 uint32_t *lock,
 uint32_t id
)
{
  // Only try finite number of times
  for (size_t i = 0; i < LOCK_NUM_TRIALS; ++i) {
    if (atomicCAS(lock, 0, id) == 0) {
      return true;
    }
  }
  return false;
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

__device__ __forceinline__
void
read_shared_memory
(
 uint32_t size,
 uint32_t ptr,
 uint8_t *buf
)
{
#if 0
// TODO(Keren): faster memory read optimization for different layers
  switch (size) {
    case 16:
      {
        uint64_t ret1, ret2;
        asm volatile("ld.shared.b64 %0,[%1];" : "=l"(ret1) : "r"(ptr) : "memory");
        asm volatile("ld.shared.b64 %0,[%1];" : "=l"(ret2) : "r"(ptr + 8) : "memory");
        uint64_t *tmp = (uint64_t *)buf;
        tmp[0] = ret1;
        tmp[1] = ret2;
        break;
      }
    case 8:
      {
        uint64_t ret;
        asm volatile("ld.shared.u64 %0,[%1];" : "=l"(ret) : "r"(ptr) : "memory");
        uint64_t *tmp = (uint64_t *)buf;
        tmp[0] = ret;
        break;
      }
    case 4:
      {
        uint32_t ret;
        asm volatile("ld.shared.b32 %0,[%1];" : "=r"(ret) : "r"(ptr) : "memory");
        uint32_t *tmp = (uint32_t *)buf;
        tmp[0] = ret;
        break;
      }
    case 2:
      {
        uint32_t ret;
        asm volatile("ld.shared.b16 %0,[%1];" : "=r"(ret) : "r"(ptr) : "memory");
        uint16_t *tmp = (uint16_t *)buf;
        tmp[0] = ret;
        break;
      }
    case 1:
      {
        uint32_t ret;
        asm volatile("ld.shared.b8 %0,[%1];" : "=r"(ret) : "r"(ptr) : "memory");
        uint8_t *tmp = (uint8_t *)buf;
        tmp[0] = ret;
        break;
      }
    default: 
      break;
  }
#endif

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


#endif
