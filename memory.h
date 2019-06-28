#ifndef HPCTOOLKIT_GPU_MEMORY_PATCH
#define HPCTOOLKIT_GPU_MEMORY_PATCH

#include <cstdint>
#include <vector_types.h>

struct MemoryAccess {
  uint64_t pc;
  uint64_t address;
  uint32_t accessSize;
  uint32_t flags;
  dim3 threadId;
  dim3 blockId;
};


struct MemoryAccessTracker {
  uint32_t currentEntry;
  uint32_t maxEntry;
  MemoryAccess* accesses;
};


#endif
