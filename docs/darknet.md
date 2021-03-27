# Darknet

## Introduction

[Darknet](https://github.com/AlexeyAB/darknet) is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

We studied Darknet version `b8c9c9d457a47d27710082c6e16206fc50af21f3`, used the `yolov4.cfg` and `yolov4-tiny.cfg` networks, and segment `dog.jpg`.

To compile darknet, we setup the following knobs in Makefile:

```bash
GPU=1
# append -lineinfo to the start of ARCH
ARCH=-lineinfo ...
# append -g to the start of CFLAGS
CFLAGS=-g ...
```

## Profiling

One can use gvprof to profile darknet directly. If GPU control flow graphs are wanted, we don't recommend using `gvprof -cfg` directly because darknet uses cuBLAS and cuDNN, loading hundreds of large binaries at runtime. 

## Optimization

- *data_flow* - *redundant values*

- *value_pattern* - *redundant zeros*