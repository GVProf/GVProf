# QMCPACK

## Introduction

[QMCPACK](https://github.com/QMCPACK/qmcpack) is an open-source production level many-body ab initio Quantum Monte Carlo code for computing the electronic structure of atoms, molecules, and solids.

We study QMCPACK version `474062068a9f6348dbf7d55be7d1bd375c24f1fe`.

There are a bunch of packages required to compiled QMCPACK, including clang, OpenMP (offloading), HDF5, FFTW, and BOOST. These packages can be installed directly via spack.

To compile QMCPACK, we pass the following variables to cmake:

```bash
CMAKE_C_COMPILER=mpicc
CMAKE_CXX_COMPILER=mpicxx
ENABLE_OFFLOAD=ON
USE_OBJECT_TARGET=ON
OFFLOAD_ARCH=<gpu-arch>
ENABLE_CUDA=1
CUDA_ARCH=<gpu-arch>
CUDA_HOST_COMPILER=`which gcc`
QMC_DATA=<path/to/qmc/data>
ENABLE_TIMERS=1
```

The following environment variables are also required:

```bash
export OMPI_CC=clang
export OMPI_CXX=clang++
```

## Profiling

First follow the instructions in `tests/performance/NiO/README` to enable and run the NiO tests. The configuration file used is `Nio-fcc-S1-dmc.xml` under the `batched_driver` folder.

At runtime, we use four worker threads (`export OMP_NUM_THREADS=4`). For a small scale run, one can adjust control variables such as `warmupSteps` to reduce execution time.

The data flow pattern can be profiled directly using gvprof. For the value pattern mode, one has to find the interesting function's names and use gvprof's whitelist to focus on these functions.

## Optimization

- *data_flow* - *redundant values*

[`MatrixDelayedUpdateCUDA.h: 627`](https://github.com/QMCPACK/qmcpack/blob/5c4776b747fefef0146765379461c6593108cf11/src/QMCWaveFunctions/Fermion/MatrixDelayedUpdateCUDA.h#L627). This line is often copying the same base pointers to the arrays on the GPU. Though this is not be a performance bottleneck for the current workload, it might be worth attention once the number of arrays increases. 