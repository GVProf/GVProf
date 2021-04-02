# Deepwave

## Introduction

[Deepwave](https://github.com/ar4/deepwave) is a wave propagation software implemented based on.

We studied deepwave version `1154692258da342accd21df02f7fa9ddd008f75f`. The input for deepwave is attached in GVProf's samples.

We first add `-lineinfo -g` to the `_make_cuda_extension` function in `setup.py`, and then add `-g` to the `_make_cpp_extension` function. Next we use `pip install .` to install deepwave. 

**Note that this pip is supposed be the pip installed by conda as we use conda across all the python samples**

To run the deepwave example in GVProf, we need to install matplotlib by `conda install matplotlib`.

## Profiling

Currently, using gvprof to profile python applications is intricate. We use HPCToolkit to profile and analyze deepwave separatedly. Please refer to the [FAQ](https://gvprof.readthedocs.io/en/latest/faq.html) page for the complete guide.

Since the default configuration of this example takes relatively long time, we can change `num_epochs` in this example to 1 and let it break after finishing the first batch. This deepwave application introduces high overhead (150-200x) than other applications (~20x) because its kernels access a millions of memory addresses with lots of gaps such that we are not able to merge all of the memory access ranges on the GPU. As a result, we will spend long time in both copying back memory addresses and updating host memories. 

For value pattern profiling, we monitor the most expensive propagate kernel using the following options.

```
LD_LIBRARY_PATH=/path/to/python/install/lib/python<version>/site-packages/torch:$LD_LIBRARY_PATH hpcrun -e gpu=nvidia,value_pattern@10000 -ck HPCRUN_SANITIZER_WHITELIST=./whitelist -ck HPCRUN_SANITIZER_KERNEL_SAMPLING_FREQUENCY=100000 python ./Deepwave_SEAM_example1.py
```

For data flow profiling, we turn on these knobs to accelerate the whole process.

```
LD_LIBRARY_PATH=/path/to/python/install/lib/python<version>/site-packages/torch:$LD_LIBRARY_PATH hpcrun -e gpu=nvidia,data_flow -ck HPCRUN_SANITIZER_READ_TRACE_IGNORE=1 -ck HPCRUN_SANITIZER_DATA_FLOW_HASH=0 -ck HPCRUN_SANITIZER_GPU_ANALYSIS_BLOCKS=1 -ck HPCRUN_SANITIZER_GPU_PATCH_RECORD_NUM=131072 python ./Deepwave_SEAM_example1.py

# this gives you extra speedup
# export OMP_NUM_THREADS=16
```

More information about accelerating data flow and value pattern profiling can be found in the [FAQ](https://gvprof.readthedocs.io/en/latest/faq.html) page

## Optimization

- *data_flow* - *redundant values*

We noted that the `propagate` kernel is highly suspicious, as it takes long time and iteratively updating itself with high redundant ratio (not 100%).

- *value_pattern* - *dense values*

After noticing the redundant accesses, we further profiled this kernel's value patterns and note that there is a dense values pattern with significant zeros. Because this kernel is iteratively updating values on the same memory addresses, if abundant zeros exist, many addresses will have the same values for each run. Therefore, we added a condition to check if these values are zero. If so, we perform simplified operations; otherwise, we perform the same computation as before.
