from collections import namedtuple
import subprocess
import os
import sys

TestCase = namedtuple('TestCase', ['path', 'command', 'options', 'spatial_read_files', 'spatial_read_reds', 'spatial_write_files', 'spatial_write_reds',
                                   'temporal_read_files', 'temporal_read_reds', 'temporal_write_files', 'temporal_write_reds', 'total', 'sampling', 'tolerate'])


def setup():
    test_cases = []
    # unit case
    test_cases.append(TestCase(path='samples/vectorAdd.f128',
                               command='./vectorAdd',
                               options=[],
                               spatial_read_files=[
                                   'spatial_read_t0.csv'],
                               spatial_read_reds=[3],
                               spatial_write_files=[
                                   'spatial_write_t0.csv'],
                               spatial_write_reds=[1],
                               temporal_read_files=[
                                   'temporal_read_t0.csv'],
                               temporal_read_reds=[0],
                               temporal_write_files=[
                                   'temporal_write_t0.csv'],
                               temporal_write_reds=[0],
                               total=[12],
                               sampling=0,
                               tolerate=0.0))
    # real cases
    test_cases.append(TestCase(path='samples/bfs',
                               command='./bfs',
                               options=['../data/graph1MW_6.txt'],
                               spatial_read_files=[
                                   'spatial_read_t0.csv'],
                               spatial_read_reds=[27707987],
                               spatial_write_files=[
                                   'spatial_write_t0.csv'],
                               spatial_write_reds=[7997516],
                               temporal_read_files=[
                                   'temporal_read_t0.csv'],
                               temporal_read_reds=[5603846],
                               temporal_write_files=[
                                   'temporal_write_t0.csv'],
                               temporal_write_reds=[0],
                               total=[52653451],
                               sampling=0,
                               tolerate=0.02))
    test_cases.append(TestCase(path='samples/backprop',
                               command='./backprop',
                               options=['65536'],
                               spatial_read_files=[
                                   'spatial_read_t0.csv'],
                               spatial_read_reds=[4194507],
                               spatial_write_files=[
                                   'spatial_write_t0.csv'],
                               spatial_write_reds=[1048623],
                               temporal_read_files=[
                                   'temporal_read_t0.csv'],
                               temporal_read_reds=[3149872],
                               temporal_write_files=[
                                   'temporal_write_t0.csv'],
                               temporal_write_reds=[0],
                               total=[19988592],
                               sampling=0,
                               tolerate=0.01))
    test_cases.append(TestCase(path='samples/backprop',
                               command='./backprop',
                               options=['65536'],
                               spatial_read_files=[
                                   'spatial_read_t0.csv'],
                               spatial_read_reds=[84039],
                               spatial_write_files=[
                                   'spatial_write_t0.csv'],
                               spatial_write_reds=[21009],
                               temporal_read_files=[
                                   'temporal_read_t0.csv'],
                               temporal_read_reds=[63058],
                               temporal_write_files=[
                                   'temporal_write_t0.csv'],
                               temporal_write_reds=[0],
                               total=[400160],
                               sampling=50,
                               tolerate=0.05))
    # stress test
    test_cases.append(TestCase(path='samples/stress',
                               command='./main',
                               options=[],
                               spatial_read_files=['spatial_read_t0.csv', 'spatial_read_t1.csv', 'spatial_read_t2.csv', 'spatial_read_t3.csv',
                                                   'spatial_read_t4.csv', 'spatial_read_t5.csv', 'spatial_read_t6.csv', 'spatial_read_t7.csv',
                                                   'spatial_read_t8.csv', 'spatial_read_t9.csv', 'spatial_read_t10.csv', 'spatial_read_t11.csv',
                                                   'spatial_read_t12.csv', 'spatial_read_t13.csv', 'spatial_read_t14.csv', 'spatial_read_t15.csv'],
                               spatial_read_reds=[8000, 8000, 8000, 8000,
                                                  8000, 8000, 8000, 8000,
                                                  8000, 8000, 8000, 8000,
                                                  8000, 8000, 8000, 8000],
                               spatial_write_files=['spatial_write_t0.csv', 'spatial_write_t1.csv', 'spatial_write_t2.csv', 'spatial_write_t3.csv',
                                                    'spatial_write_t4.csv', 'spatial_write_t5.csv', 'spatial_write_t6.csv', 'spatial_write_t7.csv',
                                                    'spatial_write_t8.csv', 'spatial_write_t9.csv', 'spatial_write_t10.csv', 'spatial_write_t11.csv',
                                                    'spatial_write_t12.csv', 'spatial_write_t13.csv', 'spatial_write_t14.csv', 'spatial_write_t15.csv'],
                               spatial_write_reds=[4000, 4000, 4000, 4000,
                                                   4000, 4000, 4000, 4000,
                                                   4000, 4000, 4000, 4000,
                                                   4000, 4000, 4000, 4000],
                               temporal_read_files=['temporal_read_t0.csv', 'temporal_read_t1.csv', 'temporal_read_t2.csv', 'temporal_read_t3.csv',
                                                    'temporal_read_t4.csv', 'temporal_read_t5.csv', 'temporal_read_t6.csv', 'temporal_read_t7.csv',
                                                    'temporal_read_t8.csv', 'temporal_read_t9.csv', 'temporal_read_t10.csv', 'temporal_read_t11.csv',
                                                    'temporal_read_t12.csv', 'temporal_read_t13.csv', 'temporal_read_t14.csv', 'temporal_read_t15.csv'],
                               temporal_read_reds=[79920000, 79920000, 79920000, 79920000,
                                                   79920000, 79920000, 79920000, 79920000,
                                                   79920000, 79920000, 79920000, 79920000,
                                                   79920000, 79920000, 79920000, 79920000],
                               temporal_write_files=['temporal_write_t0.csv', 'temporal_write_t1.csv', 'temporal_write_t2.csv', 'temporal_write_t3.csv',
                                                     'temporal_write_t4.csv', 'temporal_write_t5.csv', 'temporal_write_t6.csv', 'temporal_write_t7.csv',
                                                     'temporal_write_t8.csv', 'temporal_write_t9.csv', 'temporal_write_t10.csv', 'temporal_write_t11.csv',
                                                     'temporal_write_t12.csv', 'temporal_write_t13.csv', 'temporal_write_t14.csv', 'temporal_write_t15.csv'],
                               temporal_write_reds=[39960000, 39960000, 39960000, 39960000,
                                                    39960000, 39960000, 39960000, 39960000,
                                                    39960000, 39960000, 39960000, 39960000,
                                                    39960000, 39960000, 39960000, 39960000],
                               total=[120000000],
                               sampling=False,
                               tolerate=0.0001))

    return test_cases


def pipe_read(command):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout


def cleanup():
    pipe_read(['make', 'clean'])
    pipe_read(['make'])
    pipe_read(['rm', '-rf', 'gvprof-measurements*'])
    pipe_read(['rm', '-rf', 'gvprof-database*'])


def test(test_cases, bench):
    for test_case in test_cases:
        if bench is not None and bench != test_case.path:
            continue

        os.chdir(test_case.path)

        cleanup()
        sampling = ''
        if test_case.sampling != 0:
            sampling = 'sampling'
            pipe_read(['gvprof', '-e', 'redundancy@' +
                       str(test_case.sampling), test_case.command] + test_case.options)
        else:
            pipe_read(['gvprof', '-e', 'redundancy',
                       test_case.command] + test_case.options)

        def redundancy_compare(red_files, true_reds):
            for i, red_file in enumerate(red_files):
                red_file = 'gvprof-measurements/redundancy/' + red_file
                res = pipe_read(['tail', '-n', '1', red_file]).decode()
                red = float(res.split(',')[0])
                true_red = float(true_reds[i])
                epsilon = red if true_red == 0.0 else abs(
                    red - true_red) / true_red
                if epsilon > test_case.tolerate:
                    os.chdir('../..')
                    error = 'Error {} {}: (true: {} vs test: {})'.format(
                        test_case.path, red_file, true_red, red)
                    sys.exit(error)
                else:
                    print('Pass ' + test_case.path +
                          ' ' + red_file + ' ' + sampling)

        redundancy_compare(test_case.spatial_read_files,
                           test_case.spatial_read_reds)
        redundancy_compare(test_case.spatial_write_files,
                           test_case.spatial_write_reds)
        redundancy_compare(test_case.temporal_read_files,
                           test_case.temporal_read_reds)
        redundancy_compare(test_case.temporal_write_files,
                           test_case.temporal_write_reds)
        os.chdir('../..')


bench = None
if len(sys.argv) > 1:
    bench = str(sys.argv[1])

test(setup(), bench)
