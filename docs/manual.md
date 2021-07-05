# Manual

## Compile with Line Information

GVProf relies on debug information in binaries to attribute fine-grained value metrics on individual lines, loops, and functions. 

For GPU binaries, we recommend using `-O3 -lineinfo`.

For CPU binaries, we recommend using `-O3 -g`.

For software compiled with CMake system, usually we can edit `CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS` to add line info flags. Additionally, CUDA line info can be added through `CMAKE_CUDA_FLAGS`.

## Profile Using GVProf

The `gvprof` script automates a series of profiling and analysis processes, but supports only basic profiling features. For detailed profiling control, please refer to the next section.

```bash
gvprof -h
# Currently we offer three modes
# gvprof -v is your friend for debugging
gvprof -e <redundancy/data_flow/value_pattern> <app-name>
```

## Profile Using HPCToolkit

Using hpctoolkit to profile applications enables fine-grained control knobs, selective analysis of GPU/CPU binaries, and compatibilities with various launchers (e.g., jsrun).
We invoke `hpcrun` to profile an application twice using the same input.
In the first pass, we dump the cubins loaded at runtime and profile each kernel's running time. Then we invoke `hpcstruct` to analyze program structure and instruction dependency.
In the second pass, we instrument the cubins and invoke `redshow` redundancy analysis library to analyze measurement data.


### First pass
   
```bash
hpcrun -e gpu=nvidia <app-name>
hpcstruct <app-name>
# if '--gpucfg yes', hpcstruct will analyze the control flow graph of each GPU function and perform backward slicing, which is costly for large GPU binaries.
hpcstruct --gpucfg no hpctoolkit-<app-name>-measurements
# One can use also hpcstruct on the select GPU binaries only 
hpcstruct --gpucfg no <binary-name>
```
   
### Second pass

```bash
# Before profiling, we remove all profile data dumped in the first pass
rm -rf hpctoolkit-<app-name>-measurements/*.hpcrun

hpcrun -e gpu=nvidia,<mode> -ck <option1> -ck <option2> ... <app-name>
hpcprof -S <app-name>.hpcstruct hpctoolkit-<app-name>-measurements    
# If only some binaries are analyzed using hpcstruct,
# one has to supply the corresponding binaries' structure files
hpcprof -S <app-name>.hpcstruct -S <binary-name>.hpcstruct hpctoolkit-<app-name>-measurements    
```

### HPCToolkit separate pass

Large scale applications, such as Castro heavily use lambda functions and template functions for GPU kernels. Therefore, tools like `nsys` and `ncu` cannot efficiently correlate each kernel's execution time their names. Even though nvtx can provide some information to locate kernels, it is still not straightforward to map metrics back to source lines. Instead, we recommend using HPCToolkit, which provides an integrate calling context span CPUs and GPUs, to lookup the calling context and running time for each kernel. The following commands can be used.

```bash
hpcrun -e gpu=nvidia,pc <app-name>
hpcstruct <app-name>
hpcstruct --gpucfg no hpctoolkit-<app-name>-measurements
hpcprof -S <app-name>.hpcstruct hpctoolkit-<app-name>-measurements
hpcviewer hpctoolkit-<app-name>-measurements
```

## Control Knobs

The following fine-grained options can be passed to either gvprof or hpcrun by pointing the option name and option value with `-ck <option-name>=<option-value>`.

```bash
HPCRUN_SANITIZER_GPU_PATCH_RECORD_NUM=<size of the buffer on GPU, default: 16 * 1024>
HPCRUN_SANITIZER_BUFFER_POOL_SIZE=<size of the buffer pool on CPU, default: 500>
HPCRUN_SANITIZER_APPROX_LEVEL=<enable approximated profiling, 0-5, default: 0>
HPCRUN_SANITIZER_PC_VIEWS=<number of top redundant values per pc, default: 0>
HPCRUN_SANITIZER_MEM_VIEWS=<number of top redundant values per memory object, default: 0>
HPCRUN_SANITIZER_DEFAULT_TYPE=<default data type of memory objects, default: float>
HPCRUN_SANITIZER_KERNEL_SAMPLING_FREQUENCY=<kernel sampling frequency, default: 1>
HPCRUN_SANITIZER_WHITELIST=<functions to be monitored during execution, default: 0>
HPCRUN_SANITIZER_BLACKLIST=<functions not monitored during execution, default: 0>
HPCRUN_SANITIZER_READ_TRACE_IGNORE=<if read addresses are ignored, default: 0>
HPCRUN_SANITIZER_DATA_FLOW_HASH=<if SHA256 hash is calculated for every operation, default: 0>
HPCRUN_SANITIZER_GPU_ANALYSIS_BLOCKS=<number of gpu blocks dedicated for analysis, default: 0>
```

## Interpret Profile Data

Currently, GVProf supports using hpcviewer to associate the redundancy metrics with individual GPU source code and using gviewer to process data flow metrics and prune unnecessary nodes/edges. We plan to integrate value pattern metrics into the data flow view for more friendly use of GVProf.

### Calling context view 

Only CPU calling context is available now.
GPU calling context is under development.

```bash
hpcviewer <database-dir>
```
      
### Data flow view

```bash
gviewer -f <database-dir>/data_flow.dot.context -cf file -p 
# gviewer -h for detailed options
```
The generated .svg can be visualized directly. To enable interactive control, we can rename the file to `demo.svg` and move it to `jquery.graphviz.svg`. After launch a server locally, we can visualize the graph, zoom in for important parts, and track each node's data flows.

### Fine grain pattern views

```bash
# value pattern
less <database-dir>/value_pattern_t<cpu-thread-id>.csv

# redundancy
less <database-dir>/temporal_read_t<cpu-thread-id>.csv
less <database-dir>/temporal_write_t<cpu-thread-id>.csv
less <database-dir>/spatial_read_t<cpu-thread-id>.csv
less <database-dir>/spatial_write_t<cpu-thread-id>.csv
```

## Example

<work-in-progress>
