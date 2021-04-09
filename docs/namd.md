# NAMD

## Introduction

[NAMD](https://www.ks.uiuc.edu/Research/namd/) is a parallel molecular dynamics code designed for high-performance simulation of large biomolecular systems. NAMD uses the popular molecular graphics program VMD for simulation setup and trajectory analysis.

We download NAMD source code from its [official website](https://www.ks.uiuc.edu/Development/Download/download.cgi). We use NAMD version `4a41c6087f69c4cfe3edfdb19c6a5780ac20f5f1` and study the `alanin.namd` input.

The following flags are setup in `Make.config`:

```
CUDAGENCODE = -arch <gpu-arch> -g -lineinfo
CXX_OPTS = -g -O3            
```

## Profiling

For data flow profiling, we use the normal gvprof script with the `-ck HPCRUN_SANITIZER_READ_TRACE_IGNORE=1` option.

For value pattern profiling, we choose to monitor the most costly `nonbondedForceKernel` kernel of namd. Note that because this function involves many arrays with different value types, we need GPU control flow graph and backward slicing to derive the types of each array.
For your reference, we use the command
```
gvprof -cfg -j 16 -e value_pattern -ck HPCRUN_SANITIZER_WHITELIST=./whitelist -ck HPCRUN_SANITIZER_KERNEL_SAMPLING_FREQUENCY=10
```
The CFG analysis phase could take up to an hour consuming about **100GB** main memory.

**Caution: please use the full mangled name of `nonbondedForceKernel`**

## Optimization

- *data_flow* - *redundant values*

We found the kernels in Figure x are repetively invoked, forming an interesting diagram.
By looking carefully into the code, we found that the redundancy is introduced with purpose.
The authors of namd pays very close attention to its performance. They allocate some variables on the device to accumulate global sums and only transfer these value back to the host using the last block of the kernel. Besides, at end very end of these kernels, they reset these values to zeros to make sure the next time the buffers are clean.

You may wonder they are doing this. There are two reasons:

1. If variables are not cleaned on the device, we have to reset variable using either memsetAsync or implicit device host communication and triggers extra cost. While directly set it in a GPU kernel can hide this latency with other computations without additional API invocation.

2. If the host variable is accessed every time, these kernels will be slowed down significantly.

- *value_pattern* - *type overuse*

`CudaComputeNonbondedKernel.cu: 579`. By profiling the value patterns of this `CudaCOmputeNonbondedKernel` kernel, we found the type overuse for this memory. Therefore, we use `uint8_t` to replace the original `int` data type.