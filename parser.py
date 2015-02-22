import numpy as np
import networkx as nx

import pysoft


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

    return np.array(mat)

def parse_concentration(fname, conc_range=[0]):
    """ Return all concentrations (at specified point in time) of given entries and list of unprocessable entries
    """
    soft = pysoft.SOFTFile(fname)

    data = {}
    for row in soft.data:
        gene = row['IDENTIFIER'].lower()

        conc = []
        cont = False
        for i in conc_range:
            try:
                conc.append(float(row[2+i]))
            except (ValueError, IndexError) as e:
                cont = True
                break
        if cont:
            continue

        if len(conc) == 1: conc = conc[0] # fix for compatibility and stuff
        data[gene] = conc

    return data
