# Rodinia GPU Benchmark

## backprop

- vp-opt1: *value_pattern* - *redundant zeros*

[`backprop_cuda_kernel.cu: 81`](https://github.com/FindHao/GVProf-samples/blob/110a7cdb0d57f5902941deb59899e6266f79844e/backprop/backprop_cuda_kernel.cu#L81). The *delta* array has many zeros. We can check each entry on the GPU side to execute a special branch that avoid computation.

- vp-opt2: *data_flow* - *duplicate values*

[`backprop_cuda.cu: 180`](https://github.com/FindHao/GVProf-samples/blob/110a7cdb0d57f5902941deb59899e6266f79844e/backprop/backprop_cuda.cu#L180). *net->input_units* is copied to GPU at *Line 118* and copied back at *Line 188*. Meanwhile, both the GPU data and the CPU data are not changed. As a result, the copy at *Line 188* can be eliminated safely.

## bfs

- vp-opt1: *value_pattern* - *type overuse*

[`kernel.cu: 22`](https://github.com/FindHao/GVProf-samples/blob/110a7cdb0d57f5902941deb59899e6266f79844e/bfs/kernel.cu#L22). The *g_cost*'s array's values are within the range of `[-127, 128)`. We can specify this array's type as `int_8` instead of `int` to reduce both kernel execution time and memory copy time.

- vp-opt2: *value_pattern* - *dense values*

[`bfs.cu: 107-109`](https://github.com/FindHao/GVProf-samples/blob/110a7cdb0d57f5902941deb59899e6266f79844e/bfs/bfs.cu#L107). Accesses to these arrays showing a dense value pattern where zeros are read most of the time. We can replace the memory copies of all zeros from CPU to GPU by memset that is much faster to reduce memory copy time.

## cfd

- vp-opt1: *value_pattern* - *dense values*

[`euler3d.cu: 173`](https://github.com/FindHao/GVProf-samples/blob/110a7cdb0d57f5902941deb59899e6266f79844e/cfd/euler3d.cu#L173). The *cuda_initialize_variables* function writes values in a dense pattern. We can *hash* the accessing index of this array to limit memory access in a certain range and increase cache locality. Since this array is changed in the second iteration, this optimization only applies to the first iteration.

- vp-opt2: *data_flow* - *redundant values*

[`euler3d.cu: 570`](https://github.com/FindHao/GVProf-samples/blob/110a7cdb0d57f5902941deb59899e6266f79844e/cfd/euler3d.cu#L570). The *old_variables* array is originally initialized at *Line 551* with the same values are *variables* but copied again at *Line 570*. We can safely eliminate the second copy which is redundant to the first iteration.

## hotspot

- vp-opt: *value_pattern* - *approximate* - *single value*

[`hotspot.cu: 164`](https://github.com/FindHao/GVProf-samples/blob/110a7cdb0d57f5902941deb59899e6266f79844e/hotspot/hotspot.cu#L164). The *temp_src* array contains many very close floating point numbers.
Using the approximate mode, gvprof determines values in this array are approximately the same under a certain approximation level.
Therefore, we can read just some neighbor points on *Line 195* and still get similar final results.

## hotspot3D

- vp-opt: *value_pattern* - *approximate* - *single value*

[`opt1.cu: 29`](https://github.com/FindHao/GVProf-samples/blob/110a7cdb0d57f5902941deb59899e6266f79844e/hotspot3D/opt1.cu#L29). Like the *hotspot* example, the *tIn* array contains many very close floating point numbers. And gvprof determines all this values in this array are approximately the same under the certain approximation level. Incontrast to the *hotspot* example that selectively choose neighors, we use loop perforation to compute half of the loops and get similar result.

## huffman

- vp-opt: *value_pattern* - *dense values*

[`his.cu: 51`](https://github.com/FindHao/GVProf-samples/blob/110a7cdb0d57f5902941deb59899e6266f79844e/huffman/hist.cu#L51). GVProf reports dense values for the histo array in both the write and read modes. Because the most frequently updated value is zero, we can conditionally perform atomicAdd to reduce atomic operations.

## lavaMD

- vp-opt: *value_pattern* - *type overuse*

[`kernel_gpu_cuda.cu: 84`](https://github.com/FindHao/GVProf-samples/blob/110a7cdb0d57f5902941deb59899e6266f79844e/lavaMD/kernel/kernel_gpu_cuda.cu#L84). The *rA* array contains only few distinct numbers. By checking its initialization on the CPU side, we note that there are only ten fixed values within 0.1 to 1.0. We can store these values using `uint_8` instead of `double`, saving *8x* space. These values are then decoded on the GPU side. In this way, we trade in compute time for memory copy time.

## pathfinder

- vp-opt: *value_pattern* - *type overuse*

[`pathfinder.cu: 144`](https://github.com/FindHao/GVProf-samples/blob/110a7cdb0d57f5902941deb59899e6266f79844e/pathfinder/pathfinder.cu#L144). The *gpuWall* array's values for this input will be within `[0, 255]`, thereby we can use `uint8_t` to replace `int` to reduce global memory traffic.

## srad

- vp-opt1: *value_pattern* - *single value*

[`srad_kernel.cu: 79`](https://github.com/FindHao/GVProf-samples/blob/110a7cdb0d57f5902941deb59899e6266f79844e/srad_v1/srad_kernel.cu#L79). *d_c_loc* is always one for this output. We can memset all the values of *d_c* to 1 beforehand and to eliminate all stores with 1s

- vp-opt2: *value_pattern* - *structured*

[`srad_kernel.cu:`](https://github.com/GVProf/GVProf-samples/blob/a8c23e3aba/srad_v1-vp-opt2/srad_kernel.cu#L34) *d_iN*, *d_iS*, *d_jW*, *d_jE* are used to indicate the adjacent nodes' coordinates which have structured patterns. We removed these four arrays and replace them with the corresponding calculations.

## streamcluster

- vp-opt: *data_flow* - *redundant values*

[`streamcluster_cuda.cu:221`](https://github.com/FindHao/GVProf-samples/blob/110a7cdb0d57f5902941deb59899e6266f79844e/streamcluster/streamcluster_cuda.cu#L221). These arrays *center_table_d*, *switch_membership_d*, *p* are not changed in each iteration. Therefore, we can use flags on the CPU to detect if these arrays will be changed and only copy values if they are.
