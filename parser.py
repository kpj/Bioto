import numpy as np

import networkx as nx


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
    for tf in data:
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

def parse_concentration(names, file, conc_range=[0]):
    """ Returns concentrations (at specified point in time) of given entries and list of unprocessable entries
    """
    with open(file, 'r') as fd:
        content = fd.read()

    concs = []
    no_match = []
    data = {}
    header = None

    for line in content.split('\n'):
        parts = line.split()
        if len(line) == 0 or line[0] in '#^!':
            continue
        if not header:
            # assume first non-comment row to be header
            header = parts
            continue

        gene = parts[1]
        conc = []
        cont = False
        for i in conc_range:
            try:
                conc.append(float(parts[2+i]))
            except ValueError:
                cont = True
                break
        if cont:
            continue

        if len(conc) == 1: conc = conc[0] # small fix for compatibility and stuff

        data[gene.lower()] = conc

    for name in names:
        n = name.lower()
        try:
            concs.append(data[n])
        except KeyError:
            #print("Nothing found for", n)
            concs.append(0 if len(conc_range) == 1 else [0]*len(conc_range))
            no_match.append(n)

    return concs, no_match


if __name__ == '__main__':
    #generate_tf_gene_regulation('../data/network_tf_gene.txt')
    parse_concentration(['cyoB'], '../data/GDS4815_full.soft')
