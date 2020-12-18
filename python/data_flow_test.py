from collections import namedtuple
import subprocess
import os
import sys

import pygraphviz as pgv

TestCase = namedtuple(
    'TestCase', ['path', 'command', 'options', 'files', 'nodes', 'edges'])


def setup():
    test_cases = []
    # unit case
    test_cases.append(TestCase(path='samples/op_graph_simple',
                               command='./main',
                               options=[],
                               files=['data_flow.dot'],
                               nodes=[17],
                               edges=[20]))

    # real cases
    test_cases.append(TestCase(path='samples/bfs',
                               command='./bfs',
                               options=['../data/graph1MW_6.txt'],
                               files=['data_flow.dot'],
                               nodes=[23],
                               edges=[41]))

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


def test(test_cases, bench):
    for test_case in test_cases:
        if bench is not None and bench != test_case.path:
            continue

        os.chdir(test_case.path)
        cleanup()

        pipe_read(['gvprof', '-cfg', '-e', 'data_flow',
                   test_case.command] + test_case.options)

        # Just count the number of nodes and edges,
        # redundancy and overwrite is difficult for autotest
        for i, f in enumerate(test_case.files):
            f = 'gvprof-measurements/data_flow/' + f
            agraph = pgv.AGraph(f, strict=False)
            if len(agraph.nodes()) != test_case.nodes[i]:
                sys.exit('Error {} nodes (true: {} vs test: {})'.format(
                    test_case.path, test_case.nodes[i], len(agraph.nodes())))
            if len(agraph.edges()) != test_case.edges[i]:
                sys.exit('Error {} edges (true: {} vs test: {})'.format(
                    test_case.path, test_case.edges[i], len(agraph.edges())))
            print('Pass ' + test_case.path)

        os.chdir('../..')


bench = None
if len(sys.argv) > 1:
    bench = str(sys.argv[1])

test(setup(), bench)
