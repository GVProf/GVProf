# Darknet

## Introduction

[Deepwave](https://github.com/ar4/deepwave) is a wave propagation software implemented based on.

We studied deepwave version `1154692258da342accd21df02f7fa9ddd008f75f`. The input for deepwave is attached in GVProf's samples.

We first add `-lineinfo -g` to the `_make_cuda_extension` function in `setup.py`, and then add `-g` to the `_make_cpp_extension` function. Next we use `pip install -r requirements.txt` to install deepwave.

## Profiling

Currently, using gvprof to profile applications is not intricate. We use HPCToolkit to profile and analyze deepwave separatedly.

## Optimization

- *data_flow* - *redundant values*


- *value_pattern* - *redundant zeros*

The above problem as described.
