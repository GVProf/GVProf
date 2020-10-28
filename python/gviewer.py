
import argparse
import sys
import pygraphviz as pgv


def format_context(context, choice, known, leaf):
  ret = ''
  if choice == 'none':
      return ret
  frames = context.splitlines()
  for frame in frames[::-1]:
    line, func = frame.split('\t')
    if known is True and line.find('Unknown') == 0:
        continue
    if choice == 'path':
        func = ''
    elif choice == 'file':
        last_slash = line.rfind('/')
        if last_slash != -1:
            line = line[last_slash+1:]
    elif choice == 'func':
        line = ''
    ret = line + ' ' + func + '\l' + ret
    if leaf is True:
      break
  # escape characters
  ret = ret.replace('<', '\<')
  ret = ret.replace('>', '\>')
  return ret


def create_graph(args):
    file_path = args.file
    G = pgv.AGraph(file_path)
    G.node_attr['shape'] = 'record'

    for node in G.nodes():
        name = node.get_name()
        label = '{'
        label += '<name> ' + name + '|'
        for key, value in node.attr.items():
            if key == 'context':
                value = format_context(
                    value, args.context_filter, args.known, args.leaf)
            label += '{<' + key + '> ' + key.upper() + '|' + value + '}|'
        label = label[:-1]
        label += '}'
        node.attr['label'] = label

    for edge in G.edges():
        label = ''
        for key, value in edge.attr.items():
            label += key.upper() + ': ' + value + '\n'
        edge.attr['label'] = label

    return G


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='file name')
parser.add_argument('-cf', '--context-filter', choices=[
                    'path', 'file', 'func', 'all', 'none'], default='all', help='show part of the calling context')
parser.add_argument('-k', '--known', action='store_true', default=True,
                    help='show only known function')
parser.add_argument('-l', '--leaf', action='store_true', default=False,
                    help='show only leaf function')
parser.add_argument('-of', '--output-format',
                    choices=['svg', 'png', 'pdf'], default='svg', help='output format')
args = parser.parse_args()

G = create_graph(args)
G.layout(prog='dot')
G.draw(args.file + '.' + args.output_format)