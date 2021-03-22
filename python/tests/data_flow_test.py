from collections import namedtuple
import subprocess
import os
import sys

import pygraphviz as pgv

from test_cases import Test
from utils import pipe_read


class DataFlowTest(Test):
    Config = namedtuple('Config', ['files', 'nodes', 'edges'])

    def __init__(self, arch):
        super().__init__('DataFlowTest', arch)

    def setup(self, choices):
        for choice in choices:
            if choice == 'op_graph_simple':
                self._configs[choice] = DataFlowTest.Config(files=['data_flow.dot'],
                                                            nodes=[17],
                                                            edges=[20])
            elif choice == 'bfs':
                self._configs[choice] = DataFlowTest.Config(files=['data_flow.dot'],
                                                            nodes=[23],
                                                            edges=[41])

    def _run_impl(self, case_name, version):
        if case_name not in self._configs:
            return

        command = Test.cases[case_name].command
        options = Test.cases[case_name].options
        path = Test.cases[case_name].path

        pipe_read(['gvprof', '-cfg', '-e', 'data_flow',
                   command] + options)

        files = self._configs[case_name].files
        nodes = self._configs[case_name].nodes
        edges = self._configs[case_name].edges

        # Just count the number of nodes and edges,
        # redundancy and overwrite is difficult for autotest
        for i, f in enumerate(files):
            f = 'gvprof-measurements/data_flow/' + f
            agraph = pgv.AGraph(f, strict=False)
            correct = True
            if len(agraph.nodes()) != nodes[i]:
                print('Error {} nodes (true: {} vs test: {})'.format(
                    path, nodes[i], len(agraph.nodes())))
                correct = False
            if len(agraph.edges()) != edges[i]:
                print('Error {} edges (true: {} vs test: {})'.format(
                    path, edges[i], len(agraph.edges())))
                correct = False
            if correct is True:
                print('Pass ' + path)
