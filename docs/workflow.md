# Workflow

## Use GPU Patch

GPU Patch is built upon [Compute Sanitizer API](https://docs.nvidia.com/cuda/sanitizer-docs/index.html). 
As we are closely working with NVIDIA on this API, we will update GPU Patch to use new features as soon as the new release is available.
You can find a complete usage example of Sanitizer API in [`sanitizer-api.c`](https://github.com/HPCToolkit/hpctoolkit/blob/sanitizer/src/tool/hpcrun/gpu/nvidia/sanitizer-api.c).
Some simple samples can be found in this [repository](https://github.com/NVIDIA/compute-sanitizer-samples).

## Use RedShow with HPCToolkit

Please refer to the redshow [header file](https://github.com/GVProf/redshow/blob/master/include/redshow.h) for the complete set of interface.

If a new mode is added to GVProf, one should configure through the following redshow functions and sanitizer variables in HPCToolkit.

```
redshow_analysis_enable
redshow_output_dir_config

sanitizer_gpu_patch_type
sanitizer_gpu_patch_record_size
sanitizer_gpu_analysis_type
sanitizer_gpu_analysis_record_size
sanitizer_analysis_async
```

Currently, using a new runtime with redshow other than HPCToolkit is intricate, we will update the doc once we've gone through the whole process.

## GVProf Tests

GVProf has end-to-end tests for each analysis mode plus an unit test for instruction analysis. Therefore, if a new analysis mode is added, we suppose the developer to add a test using python to verify its correctness. 

For each analysis mode, the developer should write at least one simple case that covers most situations and collect results from samples.

We are in the process of completing the testing framework.

To run GVProf test, we use the following command at GVProf's root directory. The instruction test could fail due to the default data type used, which is acceptable.

```bash
python python/test.py -m all -a <gpu arch>
```
