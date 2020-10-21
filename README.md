# GVProf

[![DOI](https://zenodo.org/badge/194196140.svg)](https://zenodo.org/badge/latestdoi/194196140)

GVProf is a value profiler for NVIDIA GPUs to explore both temporal and spatial value redundancies in GPU-accelerated applications.

## Quick Start

```bash
git clone --recursive git@github.com:Jokeren/GVProf.git && cd GVProf
./bin/install
cd samples/vectorAdd.f32
make
../../bin/gvprof -e redundancy ./vectorAdd
```

## Documentations

- [Installation Guide](https://github.com/Jokeren/GVProf/blob/master/INSTALL.md)
- [User's Guide](https://github.com/Jokeren/GVProf/blob/master/MANUAL.md)
- [Developer's Guide]

## Papers

- Keren Zhou, Yueming Hao, John Mellor-Crummey, Xiaozhu Meng, and Xu Liu. [GVProf: A Value Profiler for GPU-based Clusters](https://sc20.supercomputing.org/presentation/?sess=sess164&id=pap359#038;id=pap359). In: *The International Conference for High Performance Computing, Networking, Storage, and Analysis* (SC), 2020
