# Darknet

## Introduction

[Darknet](https://github.com/AlexeyAB/darknet) is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

We check out Darknet version `312fd2e99a765949e468e18277d41f7992f08860`, study the `yolov4.cfg` and `yolov4-tiny.cfg` networks, and test an image `dog.jpg`.

To compile darknet, we setup the following knobs in Makefile:

```bash
GPU=1
# append -lineinfo to the start of ARCH
ARCH=-lineinfo ...
# append -g to the start of CFLAGS
CFLAGS=-g ...
```

## Profiling

For the data flow analysis, one can use gvprof to profile darknet directly. `-ck HPCRUN_SANITIZER_READ_TRACE_IGNORE=1` yields significant speedup.

For the value pattern analysis, we recommend using a whitelist to specify interesting GPU kernels and turning on block sampling and kernel samples.
In addition, if control flow graph based analysis is wanted, we don't recommend using `gvprof -cfg` directly because Darknet uses cuBLAS and cuDNN that trigger hundreds of large binaries loading at runtime.
In fact, darkent's data type is almost uniform across all kernels so that one can gain insights even without `-cfg`.

We can profile the fine grain patterns of darknet using

```bash
gvprof -e value_pattern@10 -ck HPCRUN_SANITIZER_WHITELIST=./whitelist -ck HPCRUN_SANITIZER_KERNEL_SAMPLING_FREQUENCY=20
```

In the `whitelist` file, we specify the following three kernels:

```
_Z15add_bias_kernelPfS_iiii
_Z21im2col_gpu_kernel_extiPKfiiiiiiiiiiiiPf
_Z26activate_array_mish_kernelPfiS_S_      
```

Other than a few kernels with frequent value patterns when approximation is used, we didn't find other interesting patterns.

**You may want to lookup real kernel names with `gvprof -v` or `readelf -s` since compilers may generate different names**

## Optimization

- *data_flow* - *redundant values*

`upsampling_layer.c: 91` and `convolution_kernels.cu: 559`. In the generated data flow graph, we found that the nodes annotated with the `fill_ongpu` kernel always have redundant accesses.
Because we run the inference mode only, the arrays are initialized with zeros and filled zeros again.
To optimize it, we can set up a flag for each array to indicate if it is "clean". A "clean" array shouldn't be filled zeros again.