# BarraCUDA

## Introduction

BarraCUDA is a GPU-accelerated sequence mapping software. BarraCUDA's code and sample data are open source and available at [sourceforge](http://seqbarracuda.sourceforge.net/). BarraCUDA's [FAQ page](http://seqbarracuda.sourceforge.net/faqs.html) provides useful instructions for installing and running benchmarks.

We study BarraCUDA *0.7.107h*, using the `Saccharomyces_cerevisiae.SGD1.01.50.dna_rm.toplevel.fa` sample data.

## Profiling

The input we used elapses for a short time so that we can profile it directly using the `gvprof` script.

```bash
# prepare
./bin/barracuda index sample_data/Saccharomyces_cerevisiae.SGD1.01.50.dna_rm.toplevel.fa

# data_flow
gvprof -e data_flow ./bin/barracuda aln sample_data/Saccharomyces_cerevisiae.SGD1.01.50.dna_rm.toplevel.fa sample_data/sample_reads.fastq > quicktest.sai

# value_pattern
gvprof -e value_pattern -cfg ./bin/barracuda aln sample_data/Saccharomyces_cerevisiae.SGD1.01.50.dna_rm.toplevel.fa sample_data/sample_reads.fastq > quicktest.sai
```

## Optimizations

- *data_flow* - *redundant values*

`barracuda.cu: 398`. In this function, cuda memory apis called after *Line 440* are not necessary when `number_of_sequences=0`.
In that case, zero data are transferred between CPUs and GPUs such that arrays remain the same values, but still triggering API invocation cost. 

- *value_pattern* - *frequent values*

`cuda2.cuh: 865`. This line copies all the elements from a local array to a global array, regardless of their values. While CPU's `memcpy` is fast for contiguous copy, GPU's `memcpy` is not. We observe that this copy operation involes many zeros. Therefore, we can create a `hits` array to record which positions have been updated, then only copy values at these positions.