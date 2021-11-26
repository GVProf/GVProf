import argparse
import time

from collections import namedtuple

from test_cases import Test
from utils import pipe_read, nsys_profile


class Benchmark(Test):
    # (kernel_name, is_template)
    Config = namedtuple('Config', ['kernels'])

    def __init__(self, arch, version):
        super().__init__('Benchmark', arch, version)
        self._kernel_time = dict()
        self._gpu_kernel_time = dict()
        self._gpu_mem_time = dict()
        self._time = dict()

    def setup(self, choices):
        for choice in choices:
            if choice == 'backprop':
                self._configs[choice] = Benchmark.Config(
                    kernels=[('bpnn_adjust_weights_cuda', False)])
            elif choice == 'bfs':
                self._configs[choice] = Benchmark.Config(
                    kernels=[('Kernel', False)])
            elif choice == 'cfd':
                self._configs[choice] = Benchmark.Config(kernels=[('cuda_compute_flux', True),
                                                                  ('cuda_time_step', True),
                                                                  ('cuda_compute_step_factor', True)])
            elif choice == 'hotspot':
                self._configs[choice] = Benchmark.Config(
                    kernels=[('calculate_temp', False)])
            elif choice == 'hotspot3D':
                self._configs[choice] = Benchmark.Config(
                    kernels=[('hotspotOpt1', False)])
            elif choice == 'huffman':
                self._configs[choice] = Benchmark.Config(
                    kernels=[('histo_kernel', False)])
            elif choice == 'lavaMD':
                self._configs[choice] = Benchmark.Config(
                    kernels=[('kernel_gpu_cuda', False)])
            elif choice == 'pathfinder':
                self._configs[choice] = Benchmark.Config(
                    kernels=[('dynproc_kernel', False)])
            elif choice == 'srad':
                self._configs[choice] = Benchmark.Config(
                    kernels=[('srad', False), ('srad2', False)])
            elif choice == 'streamcluster':
                self._configs[choice] = Benchmark.Config(kernels=[])

    def _run_impl(self, case_name, version):
        version_name = 'origin' if version is None else version

        def _init_time_dict(time_dict):
            if case_name not in time_dict:
                time_dict[case_name] = dict()

            if version_name not in time_dict[case_name]:
                time_dict[case_name][version_name] = 0.0
           
        _init_time_dict(self._kernel_time)
        _init_time_dict(self._gpu_kernel_time)
        _init_time_dict(self._gpu_mem_time)
        _init_time_dict(self._time)

        command = Test.cases[case_name].command
        options = Test.cases[case_name].options

        time_start = time.time()
        pipe_read([command] + options)
        time_end = time.time()
        elapse = time_end - time_start

        self._time[case_name][version_name] += elapse

        print('{}/{}: {}s'.format(case_name, version_name, elapse))

        kernel_times, gpu_kernel_time, gpu_mem_time = nsys_profile(
            [command] + options, self._configs[case_name].kernels)

        self._gpu_kernel_time[case_name][version_name] += gpu_kernel_time / (1e9)
        self._gpu_mem_time[case_name][version_name] += gpu_mem_time / (1e9)

        for kernel, kernel_time in kernel_times.items():
            self._kernel_time[case_name][version_name] += kernel_time / (1e9)
            print('{}/{}/{}: {}s'.format(case_name,
                                         version_name, kernel, kernel_time / (1e9)))
        print('{}/{}/gpu_kernel_time: {}s'.format(case_name,
              version_name, gpu_kernel_time / (1e9)))
        print('{}/{}/gpu_mem_time: {}s'.format(case_name,
              version_name, gpu_mem_time / (1e9)))

    def report(self):
        def _report_speedup(time_dict, dict_name):
            for case_name, times in time_dict.items():
                for version_name, version_time in times.items():
                    if version_name == 'origin':
                        continue
                    elif version_time != 0.0:
                        sp = time_dict[case_name]['origin'] / version_time
                        print('{}/{}/{}: {}x'.format(case_name, version_name, dict_name, sp))
        
        _report_speedup(self._time, 'time')
        _report_speedup(self._kernel_time, 'kernel_time')
        _report_speedup(self._gpu_kernel_time, 'gpu_kernel_time')
        _report_speedup(self._gpu_mem_time, 'gpu_mem_time')


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--case', help='case name')
parser.add_argument('-v', '--version', default='all', help='benchmark version')
parser.add_argument('-i', '--iterations', type=int, default=1)
parser.add_argument('-a', '--arch', choices=['sm_70', 'sm_72',
                    'sm_75', 'sm_80', 'sm_86'], default='sm_70', help='gpu arch name')
args = parser.parse_args()

if args.case is None:
    choice = ['backprop', 'bfs', 'cfd', 'hotspot', 'hotspot3D', 'huffman', 'lavaMD', 'pathfinder', 'srad', 'streamcluster']
else:
    choice = [args.case]

benchmark = Benchmark(args.arch, args.version)
benchmark.setup(choice)
benchmark.run(args.iterations)
benchmark.report()
