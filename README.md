# GVProf

[![DOI](https://zenodo.org/badge/194196140.svg)](https://zenodo.org/badge/latestdoi/194196140)
[![CodeFactor](https://www.codefactor.io/repository/github/gvprof/gvprof/badge/develop)](https://www.codefactor.io/repository/github/gvprof/gvprof/overview/develop)
[![Documentation Status](https://readthedocs.org/projects/gvprof/badge/?version=latest)](https://gvprof.readthedocs.io/en/latest/?badge=latest)


GVProf is a value profiler for NVIDIA GPUs to explore value-related inefficiencies in GPU-accelerated applications.

## Quick Start

```bash
git clone --recursive git@github.com:Jokeren/GVProf.git && cd GVProf

# Install gvprof
./bin/install

# Setup environment variables
export GVProfInstall=$(pwd)/gvprof
export PATH=${GVProfInstall}/bin:$PATH
export PATH=${GVProfInstall}/hpctoolkit/bin:$PATH
export PATH=${GVProfInstall}/redshow/bin:$PATH

# Test a sample
cd samples/vectorAdd.f32
make
gvprof -e redundancy ./vectorAdd
```

## Documentation

- [Installation Guide](https://gvprof.readthedocs.io/en/latest/install.html)
- [User's Guide](https://gvprof.readthedocs.io/en/latest/manual.html)
- [Developer's Guide](https://gvprof.readthedocs.io/en/latest/workflow.html)

## Papers

- Keren Zhou, Yueming Hao, John Mellor-Crummey, Xiaozhu Meng, and Xu Liu. [GVProf: A Value Profiler for GPU-based Clusters](https://dl.acm.org/doi/10.5555/3433701.3433819). In: *The International Conference for High Performance Computing, Networking, Storage, and Analysis* (SC), 2020
- Keren Zhou, Yueming Hao*, John Mellor-Crummey, Xiaozhu Meng, and Xu Liu. [ValueExpert: Exploring Value Patterns in GPU-accelerated Applications](https://dl.acm.org/doi/abs/10.1145/3503222.3507708). In: *Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems*(ASPLOS), 2022 (Keren and Yueming are co-first authors)
