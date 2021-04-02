# PyTorch

## Introduction

[PyTorch](https://pytorch.org/) is a popular machine learning framework.

We used PyTorch version `f5788898a928cb2489926c1a5418c94c598c361b`. We studied `resnet50`, `deepwave`, `pygcn`, and `bert` models. 

Based on PyTorch's README, we adopted the following commands to compile PyTorch from source.

```bash
spack install miniconda3

conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

conda install -c pytorch magma-cuda110

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export USE_CUDA=1
export REL_WITH_DEB_INFO=1
export MAX_JOBS=16
export USE_NINJA=OFF 
python setup.py install
```

- *resnet*

We get the `reset` example from the [pytorch benchmark](https://github.com/pytorch/benchmark/tree/master/torchbenchmark/models/resnet50) repo. 

To ease the installtion, we provide `1-spatial-convolution-model.py` and `1-spatial-convolution-unit.py` to check layer-wise and end-to-end performance.

- *deepwave*

We provide the instructions for installing deepwave here.

To ease checking the problematic kernel, we provide `2-replication-pad3d.py` script which only has a single `ReplicationPad3d` kernel.

- *bert*

We get the `reset` example from the [pytorch benchmark](https://github.com/pytorch/benchmark/tree/master/torchbenchmark/models/resnet50).

To ease checking the problematic kernel, we provide `3-embedding-unit.py` script which only has a single `Embedding` kernel.

- *pygcn*

We get [pygcn code](https://github.com/tkipf/pygcn) from github.

To ease checking the problematic kernel, we provide `4-indexing-unit.py` script which only has a single `index_put_` kernel.

## Profiling

Profiling a Python application takes extra steps than a normal application. We have a general guide to profile application in the [FAQ](https://gvprof.readthedocs.io/en/latest/faq.html) page.

## Optimization

We don't provide an automate performance testing suite for PyTorch in GVProf because recompile PyTorch for just small code changes still take long time and is a pain on low end servers. 

- *data_flow* - *redundant values*

Please refer to this [issue](https://github.com/pytorch/pytorch/issues/48539)

- *data_flow* - *redundant values* - *value_pattern* - *redundant zeros*

Please refer to these two: [issue1](https://github.com/pytorch/pytorch/issues/48889) and [issue2](https://github.com/pytorch/pytorch/issues/49663)

- *value_pattern* - *structured values*

In the pygcn model, we found structured value access pattern in the xxx kernel. We found that all its memory accesses can be ordered and ranked. This optimization is difficult to implement.