# NAMD

## Introduction

[NAMD](https://www.ks.uiuc.edu/Research/namd/) is a parallel molecular dynamics code designed for high-performance simulation of large biomolecular systems. NAMD uses the popular molecular graphics program VMD for simulation setup and trajectory analysis.

We download NAMD source code from its [official website](https://www.ks.uiuc.edu/Development/Download/download.cgi). We use NAMD version `4a41c6087f69c4cfe3edfdb19c6a5780ac20f5f1` and study the `alanin` input.

The following flags are setup in `Make.config`:

```
CUDAGENCODE = -arch <gpu-arch> -g -lineinfo
CXX_OPTS = -g -O3            
```

## Profiling

For data flow profiling, we use the normal gvprof script with the `-ck HPCRUN_SANITIZER_READ_TRACE_IGNORE=1` option.

For value pattern profiling, we monitor the most costly `nonbondedForceKernel` kernel of namd. Note that because this function accesses many arrays with different value types, we need GPU control flow graph and backward slicing to derive the types of each array.
For your reference, we use the command
```
gvprof -cfg -j 16 -e value_pattern -ck HPCRUN_SANITIZER_WHITELIST=./whitelist -ck HPCRUN_SANITIZER_KERNEL_SAMPLING_FREQUENCY=10
```
The CFG analysis phase could take up to an hour consuming about **100GB** main memory.

**Caution: please use the full mangled name of `nonbondedForceKernel`**

## Optimization

- *data_flow* - *redundant values*

We find the *submitHalf* kernels are repetively invoked, forming an interesting diagram.
Investigating carefully into the code, we find that the redundancy is introduced on purpose.
The authors of namd pay close attention to its performance. They allocate some variables on the device to accumulate global sums and only transfer these value back to the host using the last block of the kernel. Besides, at the end of these kernels, they reset these values to zeros to make sure the next time the buffers are clean.

You may wonder they are doing this. There are two reasons:

1. If variables are not cleaned on the device, we have to reset variable using either `memsetAsync` or implicit device host communication which trigger extra cost. In contract, directly set variables in a GPU kernel can hide this latency by overlapping memory latency with computations latencies without additional API invocation.

2. If the host variable is accessed every time, these kernels will be slowed down significantly.

- *value_pattern* - *type overuse*

`CudaComputeNonbondedKernel.cu: 579`. By profiling the value patterns of this `CudaCOmputeNonbondedKernel` kernel, we find this array's type is overused. We can use `uint8_t` to replace the original `int` data type.