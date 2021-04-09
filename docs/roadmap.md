# Roadmap

This document describes incoming features and release plans for GVProf. Since GVProf is a growing project, it has many components need fix and enhancement. Suggestions and feature requests are welcome. Users can post questions on Github's [discussion forum](https://github.com/GVProf/GVProf/discussions).

## Release v2.2

We plan release *v2.2* around Fall 2021, which will focus on enhancing the stability and compatibility of GVProf. Also, a few new features, such as customized memory allocator support and more accessible function filters are planned to be integrated.


- Features

    - NVTX

        Register CUPTI's NVTX callback to monitor customized memory allocators.

    - CUDA Memory Pool

        Support memory pool allocators in CUDA 11.2

- Bug Fixes

    - Function Filters
    
        Support substring match in whitelist and blacklist

    - Value Pattern Output

        Sort output arrays based on their access counts and fix weird numbers

- Deployment and Test

    - CMake

        Add CMake configurations to GVProf in addition to Makefile

    - Unittest

        Adapt python unittest package

    - Test configurations

        Adopt yaml files to configure test cases

## Pending Issues

We haven't decided when to solve the following issues.

- GViewer Website
    
    Launch a website to visualize data flow graphs.

- Fine grain pattern and data flow integration

    Use the website described before to show both fine grain patterns and data flow.

- HPCToolkit Merge

    Merge the latest HPCToolkit master into GVProf.
