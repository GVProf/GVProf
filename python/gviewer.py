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
    if known is True and (line.find('Unknown') == 0 or line.find('<unknown file>') == 0):
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


def format_graph(args):
    file_path = args.file
    G = pgv.AGraph(file_path)

    for node in G.nodes():
        for key, value in node.attr.items():
            if key == 'context':
                value = format_context(
                    value, args.context_filter, args.known, args.leaf)
                node.attr['context'] = value

    return G


def create_plain_graph(G):
    for node in G.nodes():
        name = node.get_name()
        label = '{'
        label += '<name> ' + name + '|'
        for key, value in node.attr.items():
            label += '{<' + key + '> ' + key.upper() + '|' + value + '}|'
        label = label[:-1]
        label += '}'
        node.attr['shape'] = 'record'
        node.attr['label'] = label

    for edge in G.edges():
        label = ''
        if edge.attr['edge_type'] == 'READ':
          label = 'EDGE_TYPE: READ\nMEMORY_NODE_ID: ' + str(edge.attr['memory_node_id'])
        else:
          for key, value in edge.attr.items():
            label += key.upper() + ': ' + value + '\n'
        edge.attr['label'] = label

    return G


def create_pretty_graph(G):
    G.graph_attr['bgcolor'] = '#2e3e56'
    G.graph_attr['pad'] = '0.5'

    for node in G.nodes():
        node.attr['shape'] = 'circle'
        node.attr['width'] = '0.6'
        node.attr['style'] = 'filled'
        node.attr['fillcolor'] = '#edad56'
        node.attr['color'] = '#edad56'
        node.attr['penwidth'] = '3'
        node.attr['tooltip'] = node.attr['context'].replace('\l', '&#10;')
    
    for edge in G.edges():
        edge.attr['color'] = '#fcfcfc'
        edge.attr['pendwidth'] = '2'
        edge.attr['fontname'] = 'helvetica Neue Ultra Light'

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
parser.add_argument('-pr', '--pretty', action='store_true', default=False,
                    help='tune output graph')
args = parser.parse_args()

G = format_graph(args)
if args.pretty:
    G = create_pretty_graph(G)
else:
    G = create_plain_graph(G)

G.layout(prog='dot')

#G.write(args.file + '.dot')
G.draw(args.file + '.' + args.output_format)