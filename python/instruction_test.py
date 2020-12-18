from collections import namedtuple
import subprocess
import os
import sys

TestCase = namedtuple('TestCase', ['path', 'command', 'options', 'insts'])


def setup():
    test_cases = []
    # unit case
    test_cases.append(TestCase(path='samples/op_pattern_simple',
                               command='./main',
                               options=[],
                               insts={'sm_70':
                                      ['FUNC: 18, PC: 0xd0, ACCESS_KIND: INTEGER,v:32,u:32',
                                       'FUNC: 19, PC: 0xc0, ACCESS_KIND: INTEGER,v:32,u:32',
                                       'FUNC: 20, PC: 0xf0, ACCESS_KIND: UNKNOWN,v:32,u:32',
                                       'FUNC: 21, PC: 0x250, ACCESS_KIND: FLOAT,v:64,u:64',
                                       'FUNC: 22, PC: 0xe0, ACCESS_KIND: UNKNOWN,v:64,u:64',
                                       'FUNC: 23, PC: 0xe0, ACCESS_KIND: FLOAT,v:64,u:64',
                                       'FUNC: 23, PC: 0x100, ACCESS_KIND: FLOAT,v:64,u:64'],
                                      'sm_75':
									  ['FUNC: 17, PC: 0xb0, ACCESS_KIND: INTEGER,v:32,u:32',
									   'FUNC: 18, PC: 0x20, ACCESS_KIND: INTEGER,v:64,u:32',
									   'FUNC: 18, PC: 0xe0, ACCESS_KIND: UNKNOWN,v:32,u:32',
									   'FUNC: 19, PC: 0x20, ACCESS_KIND: INTEGER,v:64,u:32',
									   'FUNC: 19, PC: 0x240, ACCESS_KIND: FLOAT,v:64,u:64',
									   'FUNC: 20, PC: 0x20, ACCESS_KIND: INTEGER,v:64,u:32',
									   'FUNC: 20, PC: 0xd0, ACCESS_KIND: UNKNOWN,v:64,u:64',
									   'FUNC: 21, PC: 0x20, ACCESS_KIND: INTEGER,v:64,u:32',
									   'FUNC: 21, PC: 0xd0, ACCESS_KIND: FLOAT,v:64,u:64',
									   'FUNC: 21, PC: 0xf0, ACCESS_KIND: FLOAT,v:64,u:64'],
                                      'sm_80':
                                      ['FUNC: 17, PC: 0xa0, ACCESS_KIND: INTEGER,v:64,u:64',
                                       'FUNC: 17, PC: 0xc0, ACCESS_KIND: INTEGER,v:32,u:32',
                                       'FUNC: 18, PC: 0x20, ACCESS_KIND: INTEGER,v:64,u:32',
                                       'FUNC: 18, PC: 0xc0, ACCESS_KIND: INTEGER,v:64,u:64',
                                       'FUNC: 18, PC: 0xe0, ACCESS_KIND: UNKNOWN,v:32,u:32',
                                       'FUNC: 19, PC: 0x20, ACCESS_KIND: INTEGER,v:64,u:32',
                                       'FUNC: 19, PC: 0xe0, ACCESS_KIND: INTEGER,v:64,u:64',
                                       'FUNC: 19, PC: 0x230, ACCESS_KIND: FLOAT,v:64,u:64',
                                       'FUNC: 20, PC: 0x20, ACCESS_KIND: INTEGER,v:64,u:32',
                                       'FUNC: 20, PC: 0xb0, ACCESS_KIND: INTEGER,v:64,u:64',
                                       'FUNC: 20, PC: 0xd0, ACCESS_KIND: UNKNOWN,v:64,u:64',
                                       'FUNC: 21, PC: 0x20, ACCESS_KIND: INTEGER,v:64,u:32',
                                       'FUNC: 21, PC: 0xb0, ACCESS_KIND: INTEGER,v:64,u:64',
                                       'FUNC: 21, PC: 0xd0, ACCESS_KIND: FLOAT,v:64,u:64',
                                       'FUNC: 21, PC: 0xf0, ACCESS_KIND: FLOAT,v:64,u:64']
                                      }))
    # real cases
    test_cases.append(TestCase(path='samples/bfs',
                               command='./bfs',
                               options=['../data/graph1MW_6.txt'],
                               insts={
                                   'sm_70':
                                   ['FUNC: 10, PC: 0xa0, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 10, PC: 0x170, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 10, PC: 0x180, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 10, PC: 0x190, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 10, PC: 0x1a0, ACCESS_KIND: UNKNOWN,v:8,u:8',
                                    'FUNC: 11, PC: 0x90, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 11, PC: 0xd0, ACCESS_KIND: UNKNOWN,v:8,u:8',
                                    'FUNC: 11, PC: 0xf0, ACCESS_KIND: INTEGER,v:32,u:32',
                                    'FUNC: 11, PC: 0x120, ACCESS_KIND: INTEGER,v:32,u:32',
                                    'FUNC: 11, PC: 0x1b0, ACCESS_KIND: INTEGER,v:32,u:32',
                                    'FUNC: 11, PC: 0x1f0, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 11, PC: 0x210, ACCESS_KIND: INTEGER,v:32,u:32',
                                    'FUNC: 11, PC: 0x290, ACCESS_KIND: INTEGER,v:32,u:32',
                                    'FUNC: 11, PC: 0x2a0, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 11, PC: 0x2b0, ACCESS_KIND: INTEGER,v:32,u:32',
                                    'FUNC: 11, PC: 0x2c0, ACCESS_KIND: INTEGER,v:32,u:32'],
                                   'sm_75':
								   ['FUNC: 10, PC: 0x70, ACCESS_KIND: INTEGER,v:64,u:64',
								    'FUNC: 10, PC: 0x80, ACCESS_KIND: INTEGER,v:8,u:8',
								    'FUNC: 10, PC: 0xc0, ACCESS_KIND: INTEGER,v:64,u:64',
								    'FUNC: 10, PC: 0xd0, ACCESS_KIND: INTEGER,v:64,u:64',
								    'FUNC: 10, PC: 0x100, ACCESS_KIND: UNKNOWN,v:8,u:8',
								    'FUNC: 10, PC: 0x110, ACCESS_KIND: INTEGER,v:64,u:64',
								    'FUNC: 10, PC: 0x120, ACCESS_KIND: UNKNOWN,v:8,u:8',
								    'FUNC: 10, PC: 0x130, ACCESS_KIND: INTEGER,v:64,u:64',
								    'FUNC: 10, PC: 0x140, ACCESS_KIND: UNKNOWN,v:8,u:8',
								    'FUNC: 10, PC: 0x150, ACCESS_KIND: UNKNOWN,v:8,u:8',
								    'FUNC: 11, PC: 0x70, ACCESS_KIND: INTEGER,v:64,u:64',
								    'FUNC: 11, PC: 0x80, ACCESS_KIND: INTEGER,v:8,u:8',
								    'FUNC: 11, PC: 0xd0, ACCESS_KIND: INTEGER,v:64,u:64',
								    'FUNC: 11, PC: 0x100, ACCESS_KIND: UNKNOWN,v:8,u:8',
								    'FUNC: 11, PC: 0x110, ACCESS_KIND: INTEGER,v:32,u:32',
								    'FUNC: 11, PC: 0x140, ACCESS_KIND: INTEGER,v:32,u:32',
								    'FUNC: 11, PC: 0x1c0, ACCESS_KIND: INTEGER,v:32,u:32',
								    'FUNC: 11, PC: 0x1d0, ACCESS_KIND: INTEGER,v:64,u:64',
								    'FUNC: 11, PC: 0x1f0, ACCESS_KIND: INTEGER,v:8,u:8',
								    'FUNC: 11, PC: 0x210, ACCESS_KIND: INTEGER,v:32,u:32',
								    'FUNC: 11, PC: 0x240, ACCESS_KIND: INTEGER,v:64,u:64',
								    'FUNC: 11, PC: 0x280, ACCESS_KIND: INTEGER,v:32,u:32',
								    'FUNC: 11, PC: 0x290, ACCESS_KIND: UNKNOWN,v:8,u:8',
								    'FUNC: 11, PC: 0x2a0, ACCESS_KIND: INTEGER,v:32,u:32',
								    'FUNC: 11, PC: 0x2b0, ACCESS_KIND: INTEGER,v:32,u:32'],
                                   'sm_80':
                                   ['FUNC: 10, PC: 0x70, ACCESS_KIND: INTEGER,v:64,u:64',
                                    'FUNC: 10, PC: 0xa0, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 10, PC: 0x150, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 10, PC: 0x180, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 10, PC: 0x190, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 10, PC: 0x1a0, ACCESS_KIND: UNKNOWN,v:8,u:8',
                                    'FUNC: 11, PC: 0x70, ACCESS_KIND: INTEGER,v:64,u:64',
                                    'FUNC: 11, PC: 0x90, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 11, PC: 0xd0, ACCESS_KIND: UNKNOWN,v:8,u:8',
                                    'FUNC: 11, PC: 0xf0, ACCESS_KIND: INTEGER,v:32,u:32',
                                    'FUNC: 11, PC: 0x120, ACCESS_KIND: INTEGER,v:32,u:32',
                                    'FUNC: 11, PC: 0x1a0, ACCESS_KIND: INTEGER,v:32,u:32',
                                    'FUNC: 11, PC: 0x1e0, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 11, PC: 0x200, ACCESS_KIND: INTEGER,v:32,u:32',
                                    'FUNC: 11, PC: 0x280, ACCESS_KIND: INTEGER,v:32,u:32',
                                    'FUNC: 11, PC: 0x290, ACCESS_KIND: INTEGER,v:8,u:8',
                                    'FUNC: 11, PC: 0x2a0, ACCESS_KIND: INTEGER,v:32,u:32',
                                    'FUNC: 11, PC: 0x2b0, ACCESS_KIND: INTEGER,v:32,u:32']
                               }))
    # real cases

    return test_cases


def pipe_read(command):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout


def cleanup(sm):
    pipe_read(['make', 'clean'])
    pipe_read(['make', 'ARCH=' + sm])


def test(test_cases, sm, bench):
    for test_case in test_cases:
        if bench is not None and bench != test_case.path:
            continue

        os.chdir(test_case.path)
        cleanup(sm)

        pipe_read(['gvprof', '-cfg', '-e', 'data_flow',
                   test_case.command] + test_case.options)

        files = os.listdir('./gvprof-measurements/structs/nvidia/')

        for f in files:
            if f.find('.inst') != -1:
                bufs = pipe_read(
                    ['redshow_parser', './gvprof-measurements/structs/nvidia/' + f]).decode('utf-8').splitlines()
                for n, buf in enumerate(bufs):
                    if buf != test_case.insts[sm][n]:
                        os.chdir('../..')
                        sys.exit('Error {} line {} (true: {} vs test: {})'.format(
                                 test_case.path, n, test_case.insts[sm][n], buf))
                print('Pass ' + test_case.path + ' ' + f)

        os.chdir('../..')


sm = 'sm_70'
bench = None
if len(sys.argv) == 2:
    sm = sys.argv[1]
elif len(sys.argv) == 3:
    sm = sys.argv[1]
    bench = str(sys.argv[2])

test(setup(), sm, bench)
