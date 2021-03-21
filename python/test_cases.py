from collections import namedtuple
import os

from utils import cleanup


class Test(object):
    Case = namedtuple('Case', ['path', 'versions', 'command', 'options'])
    cases = dict()

    # unit test cases
    cases['vectorAdd.f128'] = Case(
        path='samples/vectorAdd.f128', versions=[], command='./vectorAdd', options=[])
    cases['op_graph_simple'] = Case(
        path='samples/op_graph_simple', versions=[], command='./main', options=[])
    cases['op_pattern_simple'] = Case(
        path='samples/op_pattern_simple', versions=[], command='./main', options=[])

    # sample test cases
    cases['bfs'] = Case(path='samples/bfs', command='./bfs', versions=['vp_opt',
                                                                       'vp_opt1', 'vp_opt2', 'vp_opt3'], options=['../data/graph1MW_6.txt'])
    cases['backprop'] = Case(path='samples/backprop', command='./backprop',
                             versions=['vp_opt', 'vp_opt1', 'vp_opt2'], options=['65536'])

    def __init__(self, name, arch):
        self._name = name
        self._arch = arch
        self._configs = dict()

    def name(self):
        return self._name

    def setup(self, choices):
        pass

    def _run_impl(self, case_name):
        pass

    def run(self):
        cwd = os.getcwd()

        for case_name, case in Test.cases.items():
            if case_name not in self._configs:
                continue

            os.chdir(case.path)
            cleanup(self._arch)

            self._run_impl(case_name)

            os.chdir(cwd)
