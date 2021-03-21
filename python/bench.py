import argparse

from test_cases import Test
from utils import pipe_read


class Benchmark(Test):
    def __init__(self, arch):
        super(Benchmark).__init__('Benchmark', arch)

    def setup(self, choices):
        for choice in choices:
            self._configs[choice] = True

    def _run_impl(self, case):
        command = Test.cases[case].command
        options = Test.cases[case].options
        pipe_read([command] + options)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--case', help='case name')
parser.add_argument('-a', '--arch', choices=['sm_70', 'sm_72',
                    'sm_75', 'sm_80', 'sm_85'], default='sm_70', help='gpu arch name')
args = parser.parse_args()

choice = None if args.case is None else [args.case]

benchmark = Benchmark(args.arch)
benchmark.setup(choice)
benchmark.run()
