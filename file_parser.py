import operator, os.path

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

class GPNGenerator(object):
    """ Generate gene proximity network (GPN) via two different methods
    """

    def __init__(self, fname):
        self.data, self.max_right = self.parse_gene_proximity_file(fname)

    def _get_strand(self, start, end):
        if start < end:
            return list(range(start, end+1))

        first_half = list(range(start, self.max_right+1))
        second_half = list(range(1, end+1))
        return first_half + second_half

    def _get_terminus(self, origin):
        terminus = origin + (int(self.max_right/2))
        if terminus > self.max_right: terminus -= self.max_right

        return terminus

    def parse_gene_proximity_file(self, fname):
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

        data = list(sorted(data, key=operator.itemgetter('left')))
        max_right = max(data, key=operator.itemgetter('right'))['right']

        return data, max_right

    def generate_gene_proximity_network_circular(self, base_window):
        """ Generate GPN for given base window size over a circular genome (warp at beginning and end)
        """
        # generate graph
        graph = nx.MultiDiGraph()

        for i in range(len(self.data)):
            gene = self.data[i]
            roffset = gene['right'] + base_window

            # find genes in proximity
            proximity = []
            for j in range(i+1, len(self.data)):
                if self.data[j]['left'] > roffset: break
                proximity.append(self.data[j])
            else:
                new_roffset = roffset - self.max_right
                for k in range(len(self.data)):
                    if self.data[k]['left'] > new_roffset: break
                    proximity.append(self.data[k])

            edges = []
            for p in proximity:
                edges.append((gene['name'], p['name']))
                edges.append((p['name'], gene['name']))

            # add them to graph
            graph.add_node(gene['name'])
            graph.add_edges_from(edges)

        return graph

    def generate_gene_proximity_network_two_strands(self, base_window, origin):
        """ Don't assume a circular genome structure but divide it into two strands (cut at origin and terminus [at opposing side on circle])
        """
        # compute terminus
        terminus = self._get_terminus(origin)

        # generate graph
        graph = nx.MultiDiGraph()

        # generate strands
        pos_strand = self._get_strand(origin, terminus)
        neg_strand = self._get_strand(terminus+1, origin-1)

        def handle_strand(strand):
            """ Generate GPN form given strand
            """
            for e in self.data:
                start = e['left']
                if not start in strand: continue

                roffset = e['right'] + base_window
                if not roffset in strand: roffset = strand[-1]

                for n in self.data:
                    if n['left'] > start and n['left'] < roffset:
                        graph.add_node(n['name'])
                        graph.add_edges_from([(e['name'], n['name']), (n['name'], e['name'])])

        handle_strand(pos_strand)
        handle_strand(neg_strand)

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
    """
    soft = pysoft.SOFTFile(fname)
    gdsh = utils.GDSFormatHandler(soft, **kwargs)

    # prepare conc_range
    if conc_range is None or len(conc_range) == 0:
        conc_range = gdsh.get_useful_columns()
    else:
        tmp = []
        for c in conc_range:
            if isinstance(c, int):
                if c+2 < len(soft.columns):
                    tmp.append(soft.columns[c+2].name)
                else:
                    print('warning: could not select column %d from "%s"' % (c+3, fname))
            else:
                tmp.append(c)

        conc_range = tmp

    # generate basic structure of data to be returned
    res = utils.GDSParseResult(conc_range)
    res.add_filename(os.path.basename(fname))

    # gather data
    for row in gdsh.get_data():
        gene = row['IDENTIFIER'].lower()

        for col in conc_range:
            try:
                conc = float(row[col])
            except (ValueError, IndexError, KeyError) as e:
                continue

            res.data[col][gene] = conc
            res.add_gene(gene)

    return res
