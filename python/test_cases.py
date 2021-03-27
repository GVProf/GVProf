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
    cases['stress'] = Case(
        path='samples/stress', versions=[], command='./stress', options=[])

    # sample test cases
    cases['bfs'] = Case(path='samples/bfs', command='./bfs', versions=[
        'vp-opt1', 'vp-opt2', 'vp-opt'], options=['../data/graph1MW_6.txt'])
    cases['backprop'] = Case(path='samples/backprop', command='./backprop',
                             versions=['vp-opt1', 'vp-opt2', 'vp-opt'], options=['65536'])
    cases['cfd'] = Case(path='samples/cfd', command='./euler3d', versions=['vp-opt1',
                        'vp-opt2', 'vp-opt'], options=['../data/fvcorr.domn.097K'])
    cases['hotspot'] = Case(path='samples/hotspot', command='./hotspot', versions=['vp-opt'],
                            options=['512', '2', '2', '../data/temp_512', '../data/power_512', 'output.out'])
    cases['hotspot3D'] = Case(path='samples/hotspot3D', command='./3D', versions=['vp-opt'],
                              options=['512', '8', '100', '../data/power_512x8', '../data/temp_512x8', 'output.out'])
    cases['huffman'] = Case(path='samples/huffman', command='./pavle', versions=['vp-opt'],
                            options=['../data/test1024_H2.206587175259.in'])
    cases['lavaMD'] = Case(path='samples/lavaMD', command='./lavaMD', versions=['vp-opt'],
                           options=['-boxes1d', '30'])
    cases['particlefilter'] = Case(path='samples/particlefilter', command='./particlefilter_float', versions=['vp-opt'],
                                   options=['-x', '128', '-y', '128', '-z', '10', '-np', '1000'])
    cases['pathfinder'] = Case(path='samples/pathfinder', command='./pathfinder',
                               versions=['vp-opt'], options=['100000', '100', '20'])
    cases['srad'] = Case(path='samples/srad_v1', command='./srad',
                         versions=['vp-opt1', 'vp-opt2', 'vp-opt'], options=['10', '0.5', '502', '458'])
    cases['streamcluster'] = Case(path='samples/streamcluster', command='./sc_gpu', versions=['vp-opt'],
                                  options=['10', '20', '256', '65536', '65536', '1000', 'none', 'output.txt', '1'])

    def __init__(self, name, arch, version=None):
        self._name = name
        self._arch = arch
        self._version = version
        self._configs = dict()

    def name(self):
        return self._name

    def setup(self, choices):
        pass

    def _run_impl(self, case_name, version):
        pass

    def run(self, iterations=1):
        cwd = os.getcwd()
            
        for i in range(iterations):
            for case_name, case in Test.cases.items():
                if case_name not in self._configs:
                    continue

                os.chdir(case.path)
                if i == 0:
                    cleanup(self._arch)

                self._run_impl(case_name, None)

                os.chdir(cwd)

                if self._version is None:
                    continue
                
                for version in case.versions:
                    if version == self._version or self._version == 'all':
                        os.chdir(case.path + '-' + version)
                        if i == 0:
                            cleanup(self._arch)

                        self._run_impl(case_name, version)

                        os.chdir(cwd)
