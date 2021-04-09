# Workflow

## Use GPU Patch to collect data

## Develop an analysis pipeline

## Adopt redshow under with a different runtime

## GVProf tests

Currently, GVProf has end-to-end tests for each analysis mode plus an unit test for instruction analysis. Therefore, if a new analysis mode is added, we suppose the developer to add a test using python to verify its correctness. 

For each analysis mode, the developer should write at least one simple case that covers most situations and collect results from samples.

We are in the process of completing the testing framework.

To run GVProf test, we use the following command at GVProf's root directory. The instruction test could fail due to the default data type used, which is acceptable.

```bash
python python/test.py -m all -a <gpu arch>
```
