from collections import namedtuple
import subprocess
import os
import sys

from test_cases import Test
from utils import pipe_read


class RedundancyTest(Test):
    Config = namedtuple('Config', ['spatial_read_files', 'spatial_read_reds', 'spatial_write_files', 'spatial_write_reds',
                                   'temporal_read_files', 'temporal_read_reds', 'temporal_write_files', 'temporal_write_reds', 'total', 'sampling', 'tolerate'])

    def __init__(self, arch):
        super().__init__('RedundancyTest', arch)

    def setup(self, choices):
        for choice in choices:
            if choice == 'vectorAdd.f128':
                self._configs[choice] = RedundancyTest.Config(
                    spatial_read_files=['spatial_read_t0.csv'],
                    spatial_read_reds=[3],
                    spatial_write_files=['spatial_write_t0.csv'],
                    spatial_write_reds=[1],
                    temporal_read_files=['temporal_read_t0.csv'],
                    temporal_read_reds=[0],
                    temporal_write_files=['temporal_write_t0.csv'],
                    temporal_write_reds=[0],
                    total=[12],
                    sampling=0,
                    tolerate=0.0)
            elif choice == 'bfs':
                self._configs[choice] = RedundancyTest.Config(
                    spatial_read_files=['spatial_read_t0.csv'],
                    spatial_read_reds=[27707987],
                    spatial_write_files=['spatial_write_t0.csv'],
                    spatial_write_reds=[7997516],
                    temporal_read_files=['temporal_read_t0.csv'],
                    temporal_read_reds=[5603846],
                    temporal_write_files=['temporal_write_t0.csv'],
                    temporal_write_reds=[0],
                    total=[52653451],
                    sampling=0,
                    tolerate=0.02)
            elif choice == 'backprop':
                self._configs[choice] = [
                    RedundancyTest.Config(
                        spatial_read_files=['spatial_read_t0.csv'],
                        spatial_read_reds=[4194507],
                        spatial_write_files=['spatial_write_t0.csv'],
                        spatial_write_reds=[1048623],
                        temporal_read_files=['temporal_read_t0.csv'],
                        temporal_read_reds=[3149872],
                        temporal_write_files=['temporal_write_t0.csv'],
                        temporal_write_reds=[0],
                        total=[19988592],
                        sampling=0,
                        tolerate=0.01),
                    RedundancyTest.Config(
                        spatial_read_files=['spatial_read_t0.csv'],
                        spatial_read_reds=[84039],
                        spatial_write_files=['spatial_write_t0.csv'],
                        spatial_write_reds=[21009],
                        temporal_read_files=['temporal_read_t0.csv'],
                        temporal_read_reds=[63058],
                        temporal_write_files=['temporal_write_t0.csv'],
                        temporal_write_reds=[0],
                        total=[400160],
                        sampling=50,
                        tolerate=0.05)]

    def _run_impl(self, case_name, version):
        runs = self._configs[case_name]
        if not isinstance(runs, list):
            runs = [runs]

        command = Test.cases[case_name].command
        options = Test.cases[case_name].options
        path = Test.cases[case_name].path

        for run in runs:
            sampling = ''
            if run.sampling != 0:
                sampling = 'sampling'
                pipe_read(['gvprof', '-cfg', '-e', 'redundancy@' +
                           str(run.sampling), command] + options)
            else:
                pipe_read(['gvprof', '-cfg', '-e', 'redundancy',
                           command] + options)

            def redundancy_compare(red_files, true_reds):
                for i, red_file in enumerate(red_files):
                    red_file = 'gvprof-measurements/redundancy/' + red_file
                    res = pipe_read(['tail', '-n', '1', red_file]).decode()
                    red = float(res.split(',')[0])
                    true_red = float(true_reds[i])
                    epsilon = red if true_red == 0.0 else abs(
                        red - true_red) / true_red
                    if epsilon > run.tolerate:
                        print('Error {} {}: (true: {} vs test: {})'.format(
                            path, red_file, true_red, red))
                    else:
                        print('Pass ' + path + ' ' + red_file + ' ' + sampling)

            redundancy_compare(run.spatial_read_files, run.spatial_read_reds)
            redundancy_compare(run.spatial_write_files, run.spatial_write_reds)
            redundancy_compare(run.temporal_read_files, run.temporal_read_reds)
            redundancy_compare(run.temporal_write_files,
                               run.temporal_write_reds)
