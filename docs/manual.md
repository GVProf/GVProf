# Manual

## Compile with Line Information

GVProf relies on debug information in binaries to attribute fine-grained value metrics on individual lines, loops, and functions. 

For GPU binaries, we recommend using `-O3 -lineinfo`.

For CPU binaries, we recommend using `-O3 -g`.

## Profile Using GVProf

The `gvprof` script automates a series of profiling and analysis processes, but supports only basic profiling features. For detailed profiling control, please refer to the next section.

```bash
gvprof -h
# Currently we offer three modes
gvprof -e <redundancy/data_flow/value_pattern> <app-name>
```

## Profile Using HPCToolkit

Using hpctoolkit to profile applications enables fine-grained control knobs, selective analysis of GPU/CPU binaries, and compatibilities with various launchers (e.g., jsrun).
We invoke `hpcrun` to profile an application twice using the same input.
In the first pass, we dump the cubins loaded at runtime and profile each kernel's running time.
Then we invoke `hpcstruct` to analyze program structure and instruction dependency.
In the second pass, we instrument the cubins and invoke `redshow` redundancy analysis library to analyze measurement data.

- First pass
   
```bash
hpcrun -e gpu=nvidia <app-name>
hpcstruct <app-name>
hpcstruct --gpucfg yes hpctoolkit-<app-name>-measurements
# One can use hpcstruct on the focus GPU binaries only 
hpcstruct --gpucfg yes <binary-name>
```
   
- Second pass

```bash
hpcrun -e gpu=nvidia,<mode> -ck <option1> -ck <option2> ... <app-name>
hpcprof -S <app-name>.hpcstruct hpctoolkit-<app-name>-measurements    
# If only some binaries are analyzed using hpcstruct,
# one has to supply the corresponding binaries' structure files
hpcprof -S <app-name>.hpcstruct -S <binary-name>.hpcstruct hpctoolkit-<app-name>-measurements    
```

- Options

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

- Calling context view (does not contain GPU calling context currently)

```bash
hpcviewer <database-dir>
```
      
- Data flow view

```bash
python python/gviewer.py -f <measurement-dir>/data_flow.dot.context -cf file -p 
```

- Fine grain pattern view

```bash
less <measurement-dir>/value_pattern/value_pattern_t<cpu-thread-id>.csv
```
      
- Statistic view

```bash
less <measurement-dir>/redundancy/temporal_read_t<cpu-thread-id>.csv
less <measurement-dir>/redundancy/temporal_write_t<cpu-thread-id>.csv
less <measurement-dir>/redundancy/spatial_read_t<cpu-thread-id>.csv
less <measurement-dir>/redundancy/spatial_write_t<cpu-thread-id>.csv
```