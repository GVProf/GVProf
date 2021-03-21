import argparse

from data_flow_test import DataFlowTest
from redundancy_test import RedundancyTest
from value_pattern_test import ValuePatternTest
from instruction_test import InstructionTest
from test_cases import Test

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--case', help='case name')
parser.add_argument('-m', '--mode', choices=['data_flow', 'redundancy', 'value_pattern', 'instruction', 'all'], default='all', help='mode name')
parser.add_argument('-a', '--arch', choices=['sm_70', 'sm_72', 'sm_75', 'sm_80', 'sm_85'], default='sm_70', help='gpu arch name')
args = parser.parse_args()

tests = []

if args.mode == 'data_flow' or args.mode == 'all':
    tests.append(DataFlowTest(args.arch))

if args.mode == 'redundancy' or args.mode == 'all':
    tests.append(RedundancyTest(args.arch))

if args.mode == 'value_pattern' or args.mode == 'all':
    tests.append(ValuePatternTest(args.arch))

if args.mode == 'instruction' or args.mode == 'all':
    tests.append(InstructionTest(args.arch))

for test in tests:
    print("{}...".format(test.name()))
    if args.case is None:
        # Test all cases
        choice = Test.cases.keys()
    else:
        choice = [args.case]
    test.setup(choice)
    test.run()
