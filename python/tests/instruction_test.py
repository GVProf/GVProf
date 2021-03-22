from collections import namedtuple
import os
import sys

from test_cases import Test
from utils import pipe_read


class InstructionTest(Test):
    Config = namedtuple('Config', ['insts'])

    def __init__(self, arch):
        super().__init__('InstructionTest', arch)

    def setup(self, choices):
        for choice in choices:
            if choice == 'op_pattern_simple':
                self._configs[choice] = InstructionTest.Config(insts={
                    'sm_70':
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
                    ['FUNC: 17, PC: 0xa0, ACCESS_KIND: INTEGER,v:64,u:32',
                     'FUNC: 17, PC: 0xc0, ACCESS_KIND: INTEGER,v:32,u:32',
                     'FUNC: 18, PC: 0x20, ACCESS_KIND: INTEGER,v:64,u:32',
                     'FUNC: 18, PC: 0xc0, ACCESS_KIND: INTEGER,v:64,u:32',
                     'FUNC: 18, PC: 0xe0, ACCESS_KIND: UNKNOWN,v:32,u:32',
                     'FUNC: 19, PC: 0x20, ACCESS_KIND: INTEGER,v:64,u:32',
                     'FUNC: 19, PC: 0xe0, ACCESS_KIND: INTEGER,v:64,u:32',
                     'FUNC: 19, PC: 0x230, ACCESS_KIND: FLOAT,v:64,u:64',
                     'FUNC: 20, PC: 0x20, ACCESS_KIND: INTEGER,v:64,u:32',
                     'FUNC: 20, PC: 0xb0, ACCESS_KIND: INTEGER,v:64,u:32',
                     'FUNC: 20, PC: 0xd0, ACCESS_KIND: UNKNOWN,v:64,u:32',
                     'FUNC: 21, PC: 0x20, ACCESS_KIND: INTEGER,v:64,u:32',
                     'FUNC: 21, PC: 0xb0, ACCESS_KIND: INTEGER,v:64,u:32',
                     'FUNC: 21, PC: 0xd0, ACCESS_KIND: FLOAT,v:64,u:64',
                     'FUNC: 21, PC: 0xf0, ACCESS_KIND: FLOAT,v:64,u:64']
                })
            elif choice == 'bfs':
                self._configs[choice] = InstructionTest.Config(insts={
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
                    ['FUNC: 10, PC: 0x70, ACCESS_KIND: INTEGER,v:64,u:32',
                     'FUNC: 10, PC: 0xa0, ACCESS_KIND: INTEGER,v:8,u:8',
                     'FUNC: 10, PC: 0x150, ACCESS_KIND: INTEGER,v:8,u:8',
                     'FUNC: 10, PC: 0x180, ACCESS_KIND: INTEGER,v:8,u:8',
                     'FUNC: 10, PC: 0x190, ACCESS_KIND: INTEGER,v:8,u:8',
                     'FUNC: 10, PC: 0x1a0, ACCESS_KIND: UNKNOWN,v:8,u:8',
                     'FUNC: 11, PC: 0x70, ACCESS_KIND: INTEGER,v:64,u:32',
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
                })

    def _run_impl(self, case_name, version):
        command = Test.cases[case_name].command
        options = Test.cases[case_name].options
        path = Test.cases[case_name].path

        pipe_read(['gvprof', '-cfg', '-e', 'data_flow', command] + options)

        files = os.listdir('./gvprof-measurements/structs/nvidia/')

        insts = self._configs[case_name].insts

        for f in files:
            if f.find('.inst') != -1:
                bufs = pipe_read(
                    ['redshow_parser', './gvprof-measurements/structs/nvidia/' + f]).decode('utf-8').splitlines()

                correct = True
                for n, buf in enumerate(bufs):
                    if buf != insts[self._arch][n]:
                        print('Error {} line {} (true: {} vs test: {})'.format(
                            path, n, insts[self._arch][n], buf))
                        correct = False
                if correct is True:
                    print('Pass ' + path + ' ' + f)
