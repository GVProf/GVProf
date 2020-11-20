# GVProf

[![DOI](https://zenodo.org/badge/194196140.svg)](https://zenodo.org/badge/latestdoi/194196140)

GVProf is a value profiler for NVIDIA GPUs to explore value-related inefficiencies in GPU-accelerated applications.

## Quick Start

```bash
git clone --recursive git@github.com:Jokeren/GVProf.git && cd GVProf
./bin/install
cd samples/vectorAdd.f32
make
../../bin/gvprof -e redundancy ./vectorAdd
```

## Documentation

- [Installation Guide](https://github.com/Jokeren/GVProf/blob/master/INSTALL.md)
- [User's Guide](https://github.com/Jokeren/GVProf/blob/master/MANUAL.md)
- [Developer's Guide]

## Papers

- Keren Zhou, Yueming Hao, John Mellor-Crummey, Xiaozhu Meng, and Xu Liu. [GVProf: A Value Profiler for GPU-based Clusters](https://dl.acm.org/doi/10.5555/3433701.3433819). In: *The International Conference for High Performance Computing, Networking, Storage, and Analysis* (SC), 2020
