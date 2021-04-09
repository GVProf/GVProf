# PyTorch

## Introduction

[PyTorch](https://pytorch.org/) is a popular machine learning framework.

We use PyTorch version `f5788898a928cb2489926c1a5418c94c598c361b`. We study `resnet50`, `bert`, `deepwave` models. 

We apply the following commands to compile PyTorch from source.

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

We get the `resnet` example from the [pytorch benchmark](https://github.com/pytorch/benchmark/tree/master/torchbenchmark/models/resnet50) repo. 

To ease the installtion, we provide `1-spatial-convolution-model.py` and `1-spatial-convolution-unit.py` to check layer-wise and end-to-end performance.

- *deepwave*

We provide the instructions for installing deepwave here.

To ease checking the problematic kernel, we provide `2-replication-pad3d.py` script which only has a single `ReplicationPad3d` kernel.

- *bert*

We get the `reset` example from the [pytorch benchmark](https://github.com/pytorch/benchmark/tree/master/torchbenchmark/models/resnet50).

To ease checking the problematic kernel, we provide `3-embedding-unit.py` script which only has a single `Embedding` kernel.

## Profiling

Profiling a Python application takes extra steps than a normal application. We have a general guide to profile application in the [FAQ](https://gvprof.readthedocs.io/en/latest/faq.html) page.

An example profiling command is attached below for reference:

```bash
LD_LIBRARY_PATH=/path/to/python/install/lib/python<version>/site-packages/torch:$LD_LIBRARY_PATH hpcrun -e gpu=nvidia,data_flow -ck HPCRUN_SANITIZER_READ_TRACE_IGNORE=1 -ck HPCRUN_SANITIZER_DATA_FLOW_HASH=0 -ck HPCRUN_SANITIZER_GPU_ANALYSIS_BLOCKS=1 -ck HPCRUN_SANITIZER_GPU_PATCH_RECORD_NUM=131072 python ./<pytorch-script>.py
```

## Optimization

We don't provide an automate performance testing suite for PyTorch in GVProf because recompile PyTorch for just small code changes still take long time and is a pain on low end servers. 

- *data_flow* - *redundant values*

Please refer to this [issue](https://github.com/pytorch/pytorch/issues/48539)

- *data_flow* - *redundant values* - *value_pattern* - *redundant zeros*

Please refer to these two: [issue1](https://github.com/pytorch/pytorch/issues/48889) and [issue2](https://github.com/pytorch/pytorch/issues/49663)