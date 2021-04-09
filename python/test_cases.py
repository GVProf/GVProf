from collections import namedtuple
import os

from utils import cleanup


class Test(object):
    Case = namedtuple('Case', ['path', 'versions',
                      'command', 'options', 'cleanup'])
    cases = dict()

    # unit test cases
    cases['vectorAdd.f128'] = Case(
        path='samples/vectorAdd.f128', versions=[], command='./vectorAdd', options=[], cleanup=True)
    cases['op_graph_simple'] = Case(
        path='samples/op_graph_simple', versions=[], command='./main', options=[], cleanup=True)
    cases['op_pattern_simple'] = Case(
        path='samples/op_pattern_simple', versions=[], command='./main', options=[], cleanup=True)
    cases['stress'] = Case(path='samples/stress', versions=[],
                           command='./stress', options=[], cleanup=True)

    # sample test cases
    cases['bfs'] = Case(path='samples/bfs', command='./bfs', versions=['vp-opt1',
                        'vp-opt2', 'vp-opt'], options=['../data/graph1MW_6.txt'], cleanup=True)
    cases['backprop'] = Case(path='samples/backprop', command='./backprop', versions=[
                             'vp-opt1', 'vp-opt2', 'vp-opt'], options=['65536'], cleanup=True)
    cases['cfd'] = Case(path='samples/cfd', command='./euler3d', versions=['vp-opt1',
                        'vp-opt2', 'vp-opt'], options=['../data/fvcorr.domn.097K'], cleanup=True)
    cases['hotspot'] = Case(path='samples/hotspot', command='./hotspot', versions=['vp-opt'], options=[
                            '512', '2', '2', '../data/temp_512', '../data/power_512', 'output.out'], cleanup=True)
    cases['hotspot3D'] = Case(path='samples/hotspot3D', command='./3D', versions=['vp-opt'], options=[
                              '512', '8', '100', '../data/power_512x8', '../data/temp_512x8', 'output.out'], cleanup=True)
    cases['huffman'] = Case(path='samples/huffman', command='./pavle', versions=[
                            'vp-opt'], options=['../data/test1024_H2.206587175259.in'], cleanup=True)
    cases['lavaMD'] = Case(path='samples/lavaMD', command='./lavaMD',
                           versions=['vp-opt'], options=['-boxes1d', '30'], cleanup=True)
    cases['particlefilter'] = Case(path='samples/particlefilter', command='./particlefilter_float', versions=[
                                   'vp-opt'], options=['-x', '128', '-y', '128', '-z', '10', '-np', '1000'], cleanup=True)
    cases['pathfinder'] = Case(path='samples/pathfinder', command='./pathfinder',
                               versions=['vp-opt'], options=['100000', '100', '20'], cleanup=True)
    cases['srad'] = Case(path='samples/srad_v1', command='./srad', versions=['vp-opt1',
                         'vp-opt2', 'vp-opt'], options=['10', '0.5', '502', '458'], cleanup=True)
    cases['streamcluster'] = Case(path='samples/streamcluster', command='./sc_gpu', versions=['vp-opt'], options=[
                                  '10', '20', '256', '65536', '65536', '1000', 'none', 'output.txt', '1'], cleanup=True)

    # application cases
    cases['barracuda'] = Case(path='samples/barracuda', command='./barracuda', versions=['vp-opt'],
                              options=['aln', 'sample_data/Saccharomyces_cerevisiae.SGD1.01.50.dna_rm.toplevel.fa',
                              'sample_data/sample_reads.fastq', '>', 'quicktest.sai'], cleanup=False)

    cases['castro'] = Case(path='samples/Castro/Exec/hydro_tests/Sedov', command='Castro2d.gnu.CUDA.ex', versions=['vp-opt'],
                           options=['./inputs.2d.cyl_in_cartcoords'], cleanup=False)
    
    cases['darknet'] = Case(path='samples/darknet', command='./darknet', versions=['vp-opt'],
                            options=['detector', 'test', './cfg/coco.data', './cfg/yolov4.cfg',
                                     './yolov4.weights', 'data/dog.jpg', '-i', '0', '-thresh', '0.25'], cleanup=False)

    cases['deepwave'] = Case(path='samples/deepwave', command='./Deepwave_SEAM_example1.py', versions=['vp-opt'],
                             options=[], cleanup=False)

    cases['namd'] = Case(path='samples/NAMD/Linux-x86_64-g++', command='./namd3',
                         versions=['vp-opt'], options=['../alain'], cleanup=False)

    cases['qmcpack'] = Case(path='samples/qmcpack/workspace/NiO/dmc-a4-e48-batched_driver-DU8',
                            command='../../../build/bin/qmcpack', versions=['vp-opt'], options=['./NiO-fcc-S1-dmc.xml'], cleanup=False)

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
                if i == 0 and case.cleanup:
                    cleanup(self._arch)

                self._run_impl(case_name, None)

                os.chdir(cwd)

                if self._version is None:
                    continue

                for version in case.versions:
                    if version == self._version or self._version == 'all':
                        os.chdir(case.path + '-' + version)
                        if i == 0 and case.cleanup:
                            cleanup(self._arch)

                        self._run_impl(case_name, version)

                        os.chdir(cwd)
