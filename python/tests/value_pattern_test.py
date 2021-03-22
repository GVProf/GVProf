from collections import namedtuple
import subprocess
import os
import sys

from test_cases import Test
from utils import pipe_read


class ValuePatternTest(Test):
    Config = namedtuple('Config', ['files', 'op_counts', 'kernel_patterns'])

    def __init__(self, arch):
        super().__init__('ValuePatternTest', arch)

    def setup(self, choices):
        for choice in choices:
            if choice == 'op_pattern_simple':
                self._configs[choice] = ValuePatternTest.Config(
                    files=['value_pattern_t0.csv'],
                    op_counts=[[[500, 500, 1, 500], [500, 500, 1, 500]],
                               [[250, 250, 1, 250], [250, 250, 1, 250]],
                               [[500, 500, 500, 500], [500, 500, 1, 500]],
                               [[500, 500, 1, 500], [500, 500, 1, 500]],
                               [[500, 500, 500, 500]],
                               [[1000, 1000, 20, 1000]]],
                    kernel_patterns=[['Redundant Zeros', 'Single Value'],
                                     ['Redundant Zeros', 'Redundant Zeros'],
                                     ['Type Overuse', 'Single Value'],
                                     ['Redundant Zeros', 'Single Value'],
                                     ['Structured'], ['Dense Value']])
            elif choice == 'bfs':
                self._configs[choice] = ValuePatternTest.Config(
                    files=['value_pattern_t0.csv'],
                    op_counts=[[[5861406, 2000000, 1000014, 5861406],
                                [5999970, 5999970, 1000000, 5999970],
                                [12000000, 0, 0, 0],
                                [5999970, 32710, 2, 119381],
                                [1930703, 633664, 11, 1930703],
                                [1000000, 1000000, 1, 1000000],
                                [0, 0, 0, 0],
                                [1000000, 1000000, 1, 1000000],
                                [1930703, 999999, 1, 1930703],
                                [1930703, 999999, 11, 1930703]],
                               [[12000000, 1, 1, 12],
                                [999999, 999999, 1, 999999],
                                [999999, 999999, 1, 999999],
                                [0, 0, 0, 0],
                                [999999, 999999, 1, 999999],
                                [999999, 999999, 1, 999999],
                                [999999, 1, 1, 999999]]],
                    kernel_patterns=[['No Pattern', 'No Pattern', 'No Pattern',
                                      'No Pattern', 'Dense Value', 'Inappropriate',
                                      'Dense Value', 'Redundant Zeros', 'Single Value',
                                      'Dense Value'],
                                     ['No Pattern', 'Single Value', 'Inappropriate',
                                      'Dense Value', 'Redundant Zeros', 'Single Value',
                                      'Single Value']]
                )

    def _run_impl(self, case_name, version):
        def check(op_counts, kernel_patterns, buf: str):
            lines = buf.splitlines()
            order = -1
            count = -1
            pattern = -1
            find_pattern = False
            for n, line in enumerate(lines):
                count_line = False
                pattern_line = False
                dist_line = False
                if line.find('kernel id') != -1:
                    order += 1
                    pattern = -1
                elif line.find('array id:') != -1:
                    count = -1
                    pattern += 1
                    find_pattern = False
                elif line.find('count:') != -1:
                    count += 1
                    count_line = True
                elif line.find(' * ') != -1:
                    pattern_line = True
                elif line.find('TOP') != -1:
                    dist_line = True
                if count_line is True:
                    v = int(line.split(':')[1])
                    if op_counts[order][pattern][count] != v:
                        return False, ' line {} count error: (true: {} vs test: {})'.format(n, op_counts[order][pattern][count], v)
                elif pattern_line is True:
                    if line.find(kernel_patterns[order][pattern]):
                        find_pattern = True
                elif dist_line is True:
                    if find_pattern is False:
                        return False, ' line {} pattern error: (true: {})'.format(n, kernel_patterns[order][pattern])
            return True, ''

        command = Test.cases[case_name].command
        options = Test.cases[case_name].options
        path = Test.cases[case_name].path

        pipe_read(['gvprof', '-cfg', '-e', 'value_pattern', command] + options)

        files = self._configs[case_name].files
        op_counts = self._configs[case_name].op_counts
        kernel_patterns = self._configs[case_name].kernel_patterns

        for f in files:
            buf = pipe_read(
                ['cat', 'gvprof-measurements/value_pattern/' + f]).decode('utf-8')
            res, msg = check(op_counts, kernel_patterns, buf)
            if res is False:
                print('Error ' + path + ' ' + msg)
            else:
                print('Pass ' + path + ' ' + f)
