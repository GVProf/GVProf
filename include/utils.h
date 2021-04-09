#ifndef HPCTOOLKIT_GPU_PATCH_UTILITIES_H
#define HPCTOOLKIT_GPU_PATCH_UTILITIES_H

#include <stdint.h>

/*
 * Utility functions
 */
__device__ __forceinline__ uint32_t get_flat_block_id() {
  return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

__device__ __forceinline__ uint32_t get_flat_thread_id() {
  return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

__device__ __forceinline__ uint64_t get_unique_thread_id() {
  return get_flat_block_id() * blockDim.x * blockDim.y * blockDim.z + get_flat_thread_id();
}

__device__ __forceinline__ uint64_t get_grid_num_threads() {
  return gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
}

__device__ __forceinline__ uint64_t get_block_num_threads() {
  return blockDim.x * blockDim.y * blockDim.z;
}

__device__ __forceinline__ uint32_t get_laneid() {
  uint32_t laneid = 0;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid));
  return laneid;
}

__device__ __forceinline__ bool sample_callback(uint32_t frequency, uint32_t offset) {
  if (frequency != 0) {
    // 1  : Sample all blocks
    // >1 : Sample a portion of blocks
    return get_flat_block_id() % frequency == offset;
  }
  // Skip all blocks
  return false;
}

__device__ __forceinline__ bool is_locked(uint32_t *lock, uint32_t id) {
  uint32_t old = *lock;
  // Read the newest value
  __threadfence();
  return old == id;
}

__device__ __forceinline__ void read_shared_memory(uint32_t size, uint32_t ptr, uint8_t *buf) {
  for (uint32_t i = 0; i < size; ++i) {
    uint32_t ret = 0;
    asm volatile("ld.shared.b8 %0,[%1];" : "=r"(ret) : "r"(ptr + i) : "memory");
    buf[i] = ret;
  }
}

__device__ __forceinline__ void read_global_memory(uint32_t size, uint64_t ptr, uint8_t *buf) {
  for (uint32_t i = 0; i < size; ++i) {
    uint32_t ret = 0;
    asm volatile("ld.b8 %0,[%1];" : "=r"(ret) : "l"(ptr + i) : "memory");
    buf[i] = ret;
  }
}

__device__ __forceinline__ void read_local_memory(uint32_t size, uint32_t ptr, uint8_t *buf) {
  for (uint32_t i = 0; i < size; ++i) {
    uint32_t ret = 0;
    asm volatile("ld.local.b8 %0,[%1];" : "=r"(ret) : "r"(ptr + i) : "memory");
    buf[i] = ret;
  }
}

template <class T>
__device__ __forceinline__ T shfl(T v, uint32_t srcline, uint32_t mask = 0xFFFFFFFF) {
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

template <class T>
__device__ __forceinline__ T shfl_up(T v, uint32_t delta, uint32_t width = GPU_PATCH_WARP_SIZE,
                                     uint32_t mask = 0xFFFFFFFF) {
  T ret;
#if (__CUDA_ARCH__ >= 300)
#if (__CUDACC_VER_MAJOR__ >= 9)
  ret = __shfl_up_sync(mask, v, delta, width);
#else
  ret = __shfl_up(v, delta, width);
#endif
#endif
  return ret;
}

template <class T>
__device__ __forceinline__ T shfl_xor(T v, uint32_t lane_mask, uint32_t mask = 0xFFFFFFFF) {
  T ret;
#if (__CUDA_ARCH__ >= 300)
#if (__CUDACC_VER_MAJOR__ >= 9)
  ret = __shfl_xor_sync(mask, v, lane_mask);
#else
  ret = __shfl_xor(v, lane_mask);
#endif
#endif
  return ret;
}

__device__ __forceinline__ uint32_t ballot(int32_t predicate, uint32_t mask = 0xFFFFFFFF) {
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

__device__ __forceinline__ uint32_t bfe(uint32_t source, uint32_t bit_index) {
  uint32_t bit;
  asm volatile("bfe.u32 %0, %1, %2, %3;"
               : "=r"(bit)
               : "r"((uint32_t)source), "r"(bit_index), "r"(1));
  return bit;
}

__device__ __forceinline__ uint32_t brev(uint32_t source) {
  uint32_t dest;
  asm volatile("brev.b32 %0, %1;" : "=r"(dest) : "r"(source));
  return dest;
}

__device__ __forceinline__ uint32_t bfind(uint32_t source) {
  uint32_t bit_index;
  asm volatile("bfind.u32 %0, %1;" : "=r"(bit_index) : "r"((uint32_t)source));
  return bit_index;
}

__device__ __forceinline__ uint32_t fns(uint32_t source, uint32_t base_index) {
  uint32_t bit_index;
  asm volatile("fns.b32 %0, %1, %2, %3;" : "=r"(bit_index) : "r"(source), "r"(base_index), "r"(1));
  return bit_index;
}

template <typename T>
__device__ __forceinline__ T comparator(T x, uint32_t lane_mask, bool dir,
                                        uint32_t mask = 0xFFFFFFFF) {
  T y = shfl_xor(x, lane_mask, mask);
  return x < y == dir ? y : x;
}

template <typename T>
__device__ __forceinline__ T warp_sort(T x, uint32_t laneid) {
  x = comparator(x, 1, bfe(laneid, 1) ^ bfe(laneid, 0));  // A, sorted sequences of length 2
  x = comparator(x, 2, bfe(laneid, 2) ^ bfe(laneid, 1));  // B
  x = comparator(x, 1, bfe(laneid, 2) ^ bfe(laneid, 0));  // C, sorted sequences of length 4
  x = comparator(x, 4, bfe(laneid, 3) ^ bfe(laneid, 2));  // D
  x = comparator(x, 2, bfe(laneid, 3) ^ bfe(laneid, 1));  // E
  x = comparator(x, 1, bfe(laneid, 3) ^ bfe(laneid, 0));  // F, sorted sequences of length 8
  x = comparator(x, 8, bfe(laneid, 4) ^ bfe(laneid, 3));  // G
  x = comparator(x, 4, bfe(laneid, 4) ^ bfe(laneid, 2));  // H
  x = comparator(x, 2, bfe(laneid, 4) ^ bfe(laneid, 1));  // I
  x = comparator(x, 1, bfe(laneid, 4) ^ bfe(laneid, 0));  // J, sorted sequences of length 16
  x = comparator(x, 16, bfe(laneid, 4));                  // K
  x = comparator(x, 8, bfe(laneid, 3));                   // L
  x = comparator(x, 4, bfe(laneid, 2));                   // M
  x = comparator(x, 2, bfe(laneid, 1));                   // N
  x = comparator(x, 1, bfe(laneid, 0));                   // O, sorted sequences of length 32

  return x;
}

template <typename T>
__device__ __forceinline__ T atomic_load(const T *addr) {
  const volatile T *vaddr = addr;  // volatile to bypass cache
  __threadfence();                 // for seq_cst loads. Remove for acquire semantics.
  const T value = *vaddr;
  // fence to ensure that dependent reads are correctly ordered
  __threadfence();
  return value;
}

template <typename T>
__device__ __forceinline__ void atomic_store(T *addr, T value) {
  volatile T *vaddr = addr;  // volatile to bypass cache
  // fence to ensure that previous non-atomic stores are visible to other threads
  __threadfence();
  *vaddr = value;
}

template <typename T>
__device__ __forceinline__ void atomic_store_system(T *addr, T value) {
  volatile T *vaddr = addr;  // volatile to bypass cache
  // fence to ensure that previous non-atomic stores are visible to other threads
  __threadfence_system();
  *vaddr = value;
}

template <typename T, typename C>
__device__ __forceinline__ uint32_t map_upper_bound(T *map, T value, uint32_t len, C cmp) {
  uint32_t low = 0;
  uint32_t high = len;
  uint32_t mid = 0;
  while (low < high) {
    mid = (high - low) / 2 + low;
    if (cmp(map[mid], value)) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

template <typename T, typename C>
__device__ __forceinline__ uint32_t map_prev(T *map, T value, uint32_t len, C cmp) {
  uint32_t pos = map_upper_bound<T, C>(map, value, len, cmp);
  if (pos != 0) {
    --pos;
  } else {
    pos = len;
  }
  return pos;
}

#endif
