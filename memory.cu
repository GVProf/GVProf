#include "memory.h"

#include <sanitizer_patching.h>

extern "C" __device__ __noinline__
SanitizerPatchResult
memory_access_callback
(
 void* userdata,
 uint64_t pc,
 void* ptr,
 uint32_t accessSize,
 uint32_t flags
) 
{
  auto* pTracker = (MemoryAccessTracker*)userdata;

  uint32_t old = atomicAdd(&(pTracker->currentEntry), 1);

  if (old >= pTracker->maxEntry)
    return SANITIZER_PATCH_SUCCESS;

  MemoryAccess& access = pTracker->accesses[old];
  access.pc = pc;
  access.address = (uint64_t)(uintptr_t)ptr;
  access.accessSize = accessSize;
  access.flags = flags;
  access.threadId = threadIdx;
  access.blockId = blockIdx;

  return SANITIZER_PATCH_SUCCESS;
}
