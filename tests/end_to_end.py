from collections import namedtuple
import subprocess
import os
import sys

RedTestCase = namedtuple('RedTestCase', ['path', 'spatial_read_files', 'spatial_read_reds', 'spatial_write_files', 'spatial_write_reds', 'temporal_read_files', 'temporal_read_reds', 'temporal_write_files', 'temporal_write_reds', 'total', 'sampling', 'tolerate'])

def redundancy_setup():
    red_test_cases = []
    red_test_cases.append(RedTestCase(path='samples/bfs',
                          spatial_read_files=['spatial_read_t0.csv'],
                          spatial_read_reds=[27707987],
                          spatial_write_files=['spatial_write_t0.csv'],
                          spatial_write_reds=[7997516],
                          temporal_read_files=['temporal_read_t0.csv'],
                          temporal_read_reds=[5573898],
                          temporal_write_files=['temporal_write_t0.csv'],
                          temporal_write_reds=[0],
                          total=[52653451],
                          sampling=False,
                          tolerate=0.01))
    red_test_cases.append(RedTestCase(path='samples/backprop',
                          spatial_read_files=['spatial_read_t0.csv'],
                          spatial_read_reds=[4194507],
                          spatial_write_files=['spatial_write_t0.csv'],
                          spatial_write_reds=[1048623],
                          temporal_read_files=['temporal_read_t0.csv'],
                          temporal_read_reds=[3149872],
                          temporal_write_files=['temporal_write_t0.csv'],
                          temporal_write_reds=[0],
                          total=[19988592],
                          sampling=False,
                          tolerate=0.01))

    return red_test_cases

def value_flow_setup():
    value_flow_cases = []
    return value_flow_cases

def pipe_read(command):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout

def redundancy_test(test_cases):
    for test_case in test_cases:
        os.chdir(test_case.path)
        pipe_read(['make', 'clean'])
        pipe_read(['make'])
        if test_case.sampling is True:
            pipe_read(['bash', 'run_sampling.sh'])
        else:
            pipe_read(['bash', 'run.sh'])

        def redundancy_compare(red_files, true_reds):
            for i, red_file in enumerate(red_files):
                res = pipe_read(['tail', '-n', '1', red_file]).decode()
                red = float(res.split(',')[0])
                true_red = float(true_reds[i])
                epsilon = 0.0 if true_red == 0.0 else abs(red - true_red) / true_red
                if epsilon > test_case.tolerate:
                    os.chdir('../..')
                    error = 'Error {} {}: (true: {} vs test: {})'.format(test_case.path, red_file, true_red, red)
                    sys.exit(error)
                else:
                    print('Pass ' + test_case.path + ' ' + red_file)

        redundancy_compare(test_case.spatial_read_files, test_case.spatial_read_reds)
        redundancy_compare(test_case.spatial_write_files, test_case.spatial_write_reds)
        redundancy_compare(test_case.temporal_read_files, test_case.temporal_read_reds)
        redundancy_compare(test_case.temporal_write_files, test_case.temporal_write_reds)
        os.chdir('../..')

def value_flow_test(test_cases):
    pass

redundancy_test(redundancy_setup())

value_flow_test(value_flow_setup())

