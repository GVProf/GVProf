# BarraCUDA

## Introduction

BarraCUDA is a GPU-accelerated sequence mapping software. It's code and sample data are open source and available at [sourceforge](http://seqbarracuda.sourceforge.net/). BarraCUDA's [FAQ page](http://seqbarracuda.sourceforge.net/faqs.html) provides useful instructions for installing and running benchmarks.

We studied BarraCUDA *0.7.107h*, using the `Saccharomyces_cerevisiae.SGD1.01.50.dna_rm.toplevel.fa` sample data.

Since BarraCUDA's latest source code is not available on Github, we don't provide a copy in GVProf and only discuss implementation details in this document.

## Profiling

BarraCUDA is a relatively small scale application that can be profiled directly using the `gvprof` script.

```bash
barracuda index sample_data/Saccharomyces_cerevisiae.SGD1.01.50.dna_rm.toplevel.fa

# data_flow
gvprof -e data_flow barracuda aln sample_data/Saccharomyces_cerevisiae.SGD1.01.50.dna_rm.toplevel.fa sample_data/sample_reads.fastq > quicktest.sai

# value_pattern
gvprof -e value_pattern -cfg barracuda aln sample_data/Saccharomyces_cerevisiae.SGD1.01.50.dna_rm.toplevel.fa sample_data/sample_reads.fastq > quicktest.sai
```

## Optimizations

- *data_flow* - *redundant values*

`barracuda.cu: 398`. In this function, cuda memory apis called after *Line 440* are not necessariliy needed when `number_of_sequences=0`. In that case, no data is transferred between the CPU and the GPU such that the arrays keep the same values, but still triggering API invocation cost. 

- *value_pattern* - *dense values*

`cuda2.cuh: 865`. This line copies all the elements from a local array to a global array, regardless of their values. While CPU's `memcpy` is fast for contiguous copy, GPU's `memcpy` is not. We observe that this write operations involes many zeros. Therefore, we can create a `hits` array to record which positions have been updated, then only copy these positions.