import operator

import numpy as np
import networkx as nx

import pysoft

import utils


def parse_regulation_file(file):
    data = {}
    with open(file, 'r') as fd:
        content = fd.read()

        for line in content.split('\n'):
            if len(line) == 0 or line[0] == '#':
                continue

            parts = line.split()
            tf = parts[0].lower()
            gene = parts[1].lower()
            effect = parts[2] # -, +, +-

            if tf in data:
                data[tf].append((gene, effect))
            else:
                data[tf] = [(gene, effect)]

            if not gene in data:
                data[gene] = []

    return data

def generate_tf_gene_regulation(file):
    # parse file
    data = parse_regulation_file(file)

    # create graph
    graph = nx.DiGraph()
    for tf in sorted(data.keys()):
        source = tf.lower()

        for entry in data[tf]:
            sink = entry[0].lower()
            effect = entry[1]

            graph.add_nodes_from([source, sink])
            graph.add_edge(source, sink)

    return graph

def parse_gene_proximity_file(fname):
    """ Parse gene proximity network file and return nodes ordered by their left starting base and the maximal right end
    """
    data = []
    with open(fname, 'r') as fd:
        content = fd.read()

        for line in content.split('\n'):
            if len(line) == 0 or line.startswith('#'): continue

            parts = line.split()
            data.append({
                'name': parts[0].lower(),
                'left': int(parts[1]),
                'right': int(parts[2])
            })

    data = [e for e in sorted(data, key=operator.itemgetter('left'))]
    max_right = max(data, key=operator.itemgetter('right'))['right']

    return data, max_right

def generate_gene_proximity_network(fname, base_window):
    """ Generate gene proximity network (GPN) for given base window size
    """
    # gather raw data
    data, max_right = parse_gene_proximity_file(fname)

    # generate graph
    graph = nx.MultiDiGraph()

    for i in range(len(data)):
        gene = data[i]
        roffset = gene['right'] + base_window

        # find genes in proximity
        proximity = []
        for j in range(i+1, len(data)):
            if data[j]['left'] > roffset: break
            proximity.append(data[j])
        else:
            new_roffset = roffset - max_right
            for k in range(len(data)):
                if data[k]['left'] > new_roffset: break
                proximity.append(data[k])

        edges = []
        for p in proximity:
            edges.append((gene['name'], p['name']))
            edges.append((p['name'], gene['name']))

        # add them to graph
        graph.add_node(gene['name'])
        graph.add_edges_from(edges)

    return graph

def get_advanced_adjacency_matrix(file):
    """ Returns adjacency matrix where an activating relationship is represented with a 1,
        an inhibiting one with -1 and 0 otherwise.
    """
    def get_digit(effect):
        """ Codes effects to numbers
        """
        if effect == '+': return 1
        if effect == '-': return -1
        return 0 # this also includes unknown cases

    # parse file
    data = parse_regulation_file(file)

    # create matrix
    entries = sorted(data.keys())

    mat = []
    for row_ele in entries:
        col = []
        for col_ele in entries:
            for pair in data[col_ele]:
                gene = pair[0]
                effect = pair[1]

                if gene.lower() == row_ele.lower():
                    col.append(get_digit(effect))
                    break
            else:
                # no interaction
                col.append(0)

        mat.append(col)

    return np.array(mat).T

def parse_concentration(fname, conc_range=None, **kwargs):
    """ Return all concentrations (in specified column) of given entries
        Returned data is of form: {
            'gene_name_1': [<conc_1>, <conc_2>, ...],
            'gene_name_2': [<conc_1>, <conc_2>, ...],
            ...
        }
    """
    soft = pysoft.SOFTFile(fname)
    gdsh = utils.GDSFormatHandler(soft, **kwargs)

    if conc_range is None:
        conc_range = gdsh.get_useful_columns()

    data = {}
    for row in gdsh.get_data():
        gene = row['IDENTIFIER'].lower()

        conc = []
        cont = False
        base = 2
        for i in conc_range:
            try:
                if isinstance(i, int):
                    entry = 'null'
                    while entry == 'null':
                        entry = row[base+i]
                        if entry == 'null': base += 1
                else: # string
                    entry = row[i]

                conc.append(float(entry))
            except (ValueError, IndexError) as e:
                cont = True
                break
        if cont:
            continue

        if len(conc) == 1: conc = conc[0] # fix for compatibility and stuff
        data[gene] = conc

    return data
