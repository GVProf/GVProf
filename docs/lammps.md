# Lammps

## Introduction

We check out Lammps version `69d41dc16cd3272da8e768414d972b32a36803c1`, and test input `lammps/bench/in.lj` .

To compile lammps, we first edit `lammps/lib/kokkos/bin/nvcc_wrapper:37` that append `-lineinfo` to `cuda_args`. Then create a build directory under lammps, and use the following command lines to compile it.

```
cmake -DPKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=yes -D Kokkos_ENABLE_OPENMP=yes -D CMAKE_CXX_COMPILER=`pwd`/../lib/kokkos/bin/nvcc_wrapper ../cmake
../build/lmp -k on g 1 -sf kk -in in.lj 
```
