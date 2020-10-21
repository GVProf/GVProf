## Add debug flags to Makefile

GVProf relies on debug information in binaries to attribute fine-grained value redundancy metrics on individual lines, loops, and functions. 

For GPU binaries, we recommend using `-O3 -lineinfo`.

For CPU binaries, we recommend using `-O3 -g`.

## Profile applications

### gvprof script

The `gvprof` script includes basic profiling functions. For detailed profiling control, please refer to the next section.

```
gvprof -h
# Currently we offer two modes
gvprof -e <redundancy/value_flow> <app-name>
```

### Step-by-step profiling

We invoke `hpcrun` to profile an application twice using the same input.
In the first pass, we dump the cubins loaded at runtime and profile each kernel's running time.
Then we invoke `hpcstruct` to analyze program structure and instruction dependency.
In the second pass, we instrument the cubins and invoke `redshow` redundancy analysis library to analyze measurement data.

- First pass
   
      hpcrun -e gpu=nvidia <app-name>
      hpcstruct --gpucfg yes hpctoolkit-<app-name>-measurements
   
- Second pass

      hpcrun -e gpu=nvidia,redundancy -ck <option1> -ck <option2> ... <app-name>
      hpcprof hpctoolkit-<app-name>-measurements    

- Options

      HPCRUN_SANITIZER_GPU_PATCH_RECORD_NUM=<size of the buffer on GPU, default: 16 * 1024>
      HPCRUN_SANITIZER_BUFFER_POOL_SIZE=<size of the buffer pool on CPU, default: 500>
      HPCRUN_SANITIZER_APPROX_LEVEL=<enable approximated profiling, 0-5, default: 0>
      HPCRUN_SANITIZER_PC_VIEWS=<number of top redundant values per pc, default: 0>
      HPCRUN_SANITIZER_MEM_VIEWS=<number of top redundant values per memory object, default: 0>
      HPCRUN_SANITIZER_DEFAULT_TYPE=<default data type of memory objects, default: float>
      HPCRUN_SANITIZER_KERNEL_SAMPLING_FREQUENCY=<kernel sampling frequency, default: -1>

## Interpret profile data

- Calling context view (does not contain GPU calling context currently)

      hpcviewer <database-dir>
      
- Statistic view

      less <measurement-dir>/redundancy/temporal_read_t<cpu-thread-id>.csv
      less <measurement-dir>/redundancy/temporal_write_t<cpu-thread-id>.csv
      less <measurement-dir>/redundancy/spatial_read_t<cpu-thread-id>.csv
      less <measurement-dir>/redundancy/spatial_write_t<cpu-thread-id>.csv
