# Castro

## Introduction

Castro is an astrophysical radiation hydrodynamics simulation code based on AMReX framework.

We studied Castro version `5e0a1b9cbc259f4dd17f5453ba59808b4da5c3ab`.
We profiled Casto's `Exec/hydro_tests/Sedov` example using its `inputs.2d.cyl_in_cartcoords` input.

To compile Castro, we setup the following variables in `GNUmakefile`:

```bash
USE_CUDA=TRUE
CUDA_ARCH=TRUE
DIM=2
USE_MPI=FALSE
```

## Profiling

For a small scale run, we setup `max_step=20` in `inputs.2d.cyl_in_cartcoords`. To generate the data flow graph for Castro, along with redundancy metrics, we can use the `gvprof` script directly. For other fine-grained metrics, we can use `gvprof` if GPU control flow graphs are not required. Otherwise, we recommend using hpctoolkit to perform step-by-step profiling. 

## Optimization

- *data_flow* - *redundant values*

[`AMReX_Interp_2D_C.H: 344`](https://github.com/AMReX-Codes/amrex/blob/b7ddf2d2677fce63a567612978e01ced288dbda2/Src/AmrCore/AMReX_Interp_2D_C.H#L344). When castro invokes `cellconslin_slopes_mmlim`, which is an internal function implemented by AMREX, it performs `slope(i, j, n) *= a` operations for each output. With the `inputs.2d.cyl_in_cartcoords` input, somehow *a* is mostly 1.0. Thereby, we can reduce one load and one store for each output if we conditionally perform `slope(i, j, n) *= a`. Though it’s not a significant speedup, it is worth mentioning here if this optimization also benefits other applications that use AMReX.