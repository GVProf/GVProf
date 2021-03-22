# rodinia GPU benchmark

## backprop

- vp-opt1

*value_pattern*: `backprop_cuda_kernel.cu: 81`. The *delta* array has all elements zeros. We can either check the whole on the CPU side to invoke a special kernel or check each entry on the GPU side to execute a specific branch. 

- vp-opt2

*data_flow*: `backprop_cuda.cu: 188`. *net->input_units* is copied to GPU at *Line 118* and copied back at *Line 188*. Meanwhile, the both the GPU data and the CPU data are not changed. As a result, the copy at *Line 188* can be eliminated safely.
