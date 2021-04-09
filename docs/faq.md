# FAQ

- How do I profile a Python application?

Please first refer to the MANUAL page for step-by-step profiling using HPCToolkit.

In addition to the basic commands there, we also have to pay attention to other minor issues. 

In the measurement stage, `LD_LIBRARY_PATH=/path/to/python/library/:$LD_LIBRARY_PATH` may be needed as a prefix before *hpcrun*. We have a detailed example for [profiling PyTorch](https://gvprof.readthedocs.io/en/latest/deepwave.html).

Then, after getting measurement data and GPU binaries, we will analyze cpu binaries to get necessary line information.
For GPU binaries, we use *hpcstruct --gpucfg no* on the measurement directory as suggested by the manual.
For CPU binaries, the *python* binary does not contain all the program structure we need to understand program contexts.
Instead, we have to analyze these binaries loaded dynamically at runtime.
A python application may load hundreds of libraries at runtime but not use all of them.
Therefore, in order to use hpcstrut on a minimum set of binaries but still extract information to understand program contexts, we adopt a *test-and-analyze* strategy.
Using this strategy, we try hpcprof to correlate performance data with line maps first, if hpcprof hangs because of the large size of line map in a binary, we kill hpcprof and use hpcstruct on this binary to enjoy its fine-grained and fast binary analysis against hpcprof. 

When hpcprof begins analyze a binary, it will print out some message like below. In such a case, we can kill hpcprof, remove the temporary database, and use `hpcstruct` to analyze `libtorch_python.so`.

```bash
msg: Begin analyzing : /path/to/python/lib/python3.8/site-packages/torch/lib/libtorch_python.so
```

- How do I accelerate data flow profiling?

The following three knobs are helpful for accelerating proiling of applications with many kernels. With all the options turned on, the expected end-to-end of GVProf is approximately 20x, while the overhead could be over 1200x without these knobs. 

```bash
HPCRUN_SANITIZER_READ_TRACE_IGNORE=<if read addresses are ignored, default: 0>
HPCRUN_SANITIZER_DATA_FLOW_HASH=<if SHA256 hash is calculated for every operation, default: 0>
HPCRUN_SANITIZER_GPU_ANALYSIS_BLOCKS=<number of gpu blocks dedicated for analysis, default: 0>
```

Note that these knobs can disable some information generation.

When GPU analysis is enabled, one can adjust the number of records on the GPU side to enlarge the buffer on the GPU side and further reduce overhead.

```bash
HPCRUN_SANITIZER_GPU_PATCH_RECORD_NUM=<size of the buffer on GPU, default: 16 * 1024>
```

- How do I accelerate value pattern profiling?

The following knobs are helpful for profiling the value pattern of specific kernels, focusing on just several kernel instances.

Besides, one can also apply `<pattern-name>@N` to activate block sampling that profiles a random GPU block out of every *N* blocks.

```bash
HPCRUN_SANITIZER_KERNEL_SAMPLING_FREQUENCY=<kernel sampling frequency, default: 1>
HPCRUN_SANITIZER_WHITELIST=<functions to be monitored during execution, default: 0>
HPCRUN_SANITIZER_BLACKLIST=<functions not monitored during execution, default: 0>
```