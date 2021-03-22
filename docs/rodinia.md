# Rodinia GPU Benchmark

## backprop

- vp-opt1

*value_pattern*: `backprop_cuda_kernel.cu: 81`. The *delta* array has all elements zeros. We can either check the whole on the CPU side to invoke a special kernel or check each entry on the GPU side to execute a specific branch. 

- vp-opt2

*data_flow*: `backprop_cuda.cu: 188`. *net->input_units* is copied to GPU at *Line 118* and copied back at *Line 188*. Meanwhile, the both the GPU data and the CPU data are not changed. As a result, the copy at *Line 188* can be eliminated safely.

## bfs

- vp-opt1

*value_pattern*: `kernel.cu: 22`. The *g_cost*'s values are the range of [-127, 128). We can specify this array's type as `int_8` instead of `int` to reduce both kernel execution time and memory copy time.

- vp-opt2

*value_pattern*: `bfs.cu: 107-109`. Accesses to these arrays showing a dense value pattern where zero is read most of the time. We can replace the memory copies of all zeros to from CPU to GPU by memset that is way much faster to reduce memory copy time.