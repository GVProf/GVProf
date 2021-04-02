# FAQ

1. How do I profile a Python application?


2. How do I choose GPU binaries to analyze?


3. How do I accelerate data flow profiling?

The following three knobs are helpful for proiling applications with many kernels. With all the options turned on, the expected end-to-end of GVProf is approximately 20x, while the overhead could be 2000x without these knobs. 

Note that these knobs disable can disable some information generation.

```bash
HPCRUN_SANITIZER_READ_TRACE_IGNORE=<if read addresses are ignored, default: 0>
HPCRUN_SANITIZER_DATA_FLOW_HASH=<if SHA256 hash is calculated for every operation, default: 0>
HPCRUN_SANITIZER_GPU_ANALYSIS_BLOCKS=<number of gpu blocks dedicated for analysis, default: 0>
```

When GPU analysis is used, one can adjust the number of records on the GPU side to enlarge the buffer on the GPU side and further reduce overhead.

```bash
HPCRUN_SANITIZER_GPU_PATCH_RECORD_NUM=<size of the buffer on GPU, default: 16 * 1024>
```

4. How do I accelerate value pattern profiling?