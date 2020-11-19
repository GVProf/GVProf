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
    G = pgv.AGraph(file_path, strict=False)

    for node in G.nodes():
        for key, value in node.attr.items():
            if key == 'context':
                value = format_context(
                    value, args.context_filter, args.known, args.leaf)
                node.attr['context'] = value

    return G


def prune_graph(G):
    # 1. prune no edge nodes
    nodes_exist = dict()
    for node in G.nodes():
        nodes_exist[node] = False

    for edge in G.edges():
        nodes_exist[edge[0]] = True
        nodes_exist[edge[1]] = True

    for k, v in nodes_exist.items():
        if v == False:
            G.delete_node(k)

    # 2. prune low importance nodes
    # reserve important edges
    RED_THRESHOLD = 0.3
    nodes_reserve = dict()
    for edge in G.edges():
        if float(edge.attr['redundancy']) > RED_THRESHOLD:
            nodes_reserve[edge[0]] = True
            nodes_reserve[edge[1]] = True

    total_count = 0
    for node in G.nodes():
        total_count += float(node.attr['count'])

    nodes_importance = dict()
    for node in G.nodes():
        nodes_importance[node] = float(node.attr['count']) / total_count

    COUNT_THRESHOLD = 0.01
    for k, v in nodes_importance.items():
        if (k not in nodes_reserve) and (v < COUNT_THRESHOLD):
            G.delete_node(k)

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
          label = 'EDGE_TYPE: READ\nMEMORY_NODE_ID: ' + \
              str(edge.attr['memory_node_id'])
        else:
          for key, value in edge.attr.items():
            label += key.upper() + ': ' + value + '\n'
        edge.attr['label'] = label

    return G


def create_pretty_graph(G):
    def color_edge_redundancy(G):
        for edge in G.edges():
            if float(edge.attr['redundancy']) <= 0.33:
                edge.attr['color'] = '#cddc39'
                edge.attr['fillcolor'] = '#cddc39'
            elif float(edge.attr['redundancy']) <= 0.66:
                edge.attr['color'] = '#fffa55'
                edge.attr['fillcolor'] = '#fffa55'
            elif float(edge.attr['redundancy']) <= 0.99:
                edge.attr['color'] = '#fdcc3a'
                edge.attr['fillcolor'] = '#fdcc3a'
            else:
                edge.attr['color'] = '#f91100'
                edge.attr['fillcolor'] = '#f91100'
        return G

    def apportion_edge_width(G):
        edges = G.edges()
        max_edge = max(edges, key=lambda edge: float(
            edge.attr['overwrite']) * float(edge.attr['count']))
        max_width = 6.0
        max_weight = float(max_edge.attr['overwrite']) * \
            float(max_edge.attr['count'])

        for edge in edges:
            width = float(edge.attr['overwrite']) * \
                float(edge.attr['count']) / max_weight * max_width
            if width < 1.0:
                edge.attr['penwidth'] = 1.0
            else:
                edge.attr['penwidth'] = width

        return G

    def apportion_node_width(G):
        nodes = G.nodes()
        max_node = max(nodes, key=lambda node: float(node.attr['count']))
        max_width = 1.2
        max_weight = float(max_node.attr['count'])

        for node in nodes:
            width = float(node.attr['count']) / max_weight * max_width
            if width < 1.0:
                node.attr['width'] = 0.6
            else:
                node.attr['width'] = width

        return G

    def label_node_duplicate(node):
        dup = node.attr['duplicate']
        dup_entries = dup.split(';')
        from_node = node.get_name()
        label = ''
        for dup_entry in dup_entries:
            if len(dup_entry) > 0:
                dup_node = dup_entry.split(',')[0]
                label += dup_node + ' '
        return 'DUPLICATE: ' + label

    #G.graph_attr['bgcolor'] = '#2e3e56'
    G.graph_attr['pad'] = '0.5'

    for node in G.nodes():
        if node.attr['node_type'] == 'MEMORY':
            node.attr['shape'] = 'box'
        elif node.attr['node_type'] == 'KERNEL':
            node.attr['shape'] = 'ellipse'
        elif node.attr['node_type'] == 'MEMCPY' or node.attr['node_type'] == 'MEMSET':
            node.attr['shape'] = 'circle'
        else:
            node.attr['shape'] = 'box'
            node.attr['label'] = node.attr['node_type']
        node.attr['style'] = 'filled'
        node.attr['penwidth'] = '0'
        tooltip = ''
        tooltip += 'TYPE: ' + node.attr['node_type'] + '\l'
        tooltip += 'COUNT: ' + node.attr['count'] + '\l'
        duplicate = label_node_duplicate(node)
        if duplicate != '':
            tooltip += duplicate + '\l'
        tooltip += 'CONTEXT: \l' + node.attr['context']
        tooltip.replace('\l', '&#10;')
        node.attr['tooltip'] = tooltip

    # Combine read write edges
    rw_edges = dict()
    for edge in G.edges():
        if (edge[0], edge[1], edge.attr['memory_node_id']) in rw_edges:
            rw_edge = rw_edges[(edge[0], edge[1], edge.attr['memory_node_id'])]
            redundancy = float(rw_edge[1]) + float(edge.attr['redundancy'])
            overwrite = float(rw_edge[2]) + float(edge.attr['overwrite'])
            count = int(rw_edge[3]) + int(edge.attr['count'])
            rw_edges[(edge[0], edge[1], edge.attr['memory_node_id'])] = (
                True, str(redundancy), str(overwrite), str(count))
        else:
            rw_edges[(edge[0], edge[1], edge.attr['memory_node_id'])] = (
                False, edge.attr['redundancy'], edge.attr['overwrite'], edge.attr['count'])
    for edge, rw in rw_edges.items():
        if rw[0]:
            G.delete_edge(edge[0], edge[1])

    for edge in G.edges():
        tooltip = 'MEMORY_NODE_ID: ' + edge.attr['memory_node_id'] + '\l'
        if rw_edges[edge[0], edge[1], edge.attr['memory_node_id']][0]:
            rw_edge = rw_edges[(edge[0], edge[1], edge.attr['memory_node_id'])]
            tooltip += 'TYPE: READ \& WRITE' + '\l'
            tooltip += 'REDUNDANCY: ' + str(rw_edge[1]) + '\l'
            tooltip += 'OVERWRITE: ' + str(rw_edge[2]) + '\l'
            tooltip += 'BYTES: ' + str(rw_edge[3]) + '\l'
        else:
            tooltip += 'TYPE: ' + edge.attr['edge_type'] + '\l'
            tooltip += 'REDUNDANCY: ' + str(edge.attr['redundancy']) + '\l'
            tooltip += 'OVERWRITE: ' + str(edge.attr['overwrite']) + '\l'
            tooltip += 'BYTES: ' + str(edge.attr['count']) + '\l'
        tooltip.replace('\l', '&#10;')
        edge.attr['tooltip'] = tooltip
        edge.attr['fontname'] = 'helvetica Neue Ultra Light'

    G = apportion_node_width(G)

    G = color_edge_redundancy(G)

    G = apportion_edge_width(G)

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
parser.add_argument('-p', '--prune', action='store_true', default=False)
parser.add_argument(
    '-ly', '--layout', choices=['dot', 'neato', 'circo'], default='dot', help='svg layout')
parser.add_argument('-pr', '--pretty', action='store_true', default=False,
                    help='tune output graph')
parser.add_argument('-v', '--verbose', action='store_true', help='print log')
args = parser.parse_args()

if args.verbose:
  print('Format graph...')
G = format_graph(args)

if args.prune:
  if args.verbose:
    print('Prune  graph: {} nodes and {} edges...'.format(
        len(G.nodes()), len(G.edges())))
  G = prune_graph(G)

if args.verbose:
  print('Refine graph...')
if args.pretty:
    G = create_pretty_graph(G)
else:
    G = create_plain_graph(G)

if args.verbose:
  print('Organize graph: {} nodes and {} edges...'.format(
      len(G.nodes()), len(G.edges())))
G.layout(prog=args.layout)

if args.verbose:
  print('Output graph...')
#G.write(args.file + '.dot')
G.draw(args.file + '.' + args.output_format)
