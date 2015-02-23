import os, os.path
import json, csv
import hashlib
import random
import operator

import numpy as np
import numpy.random as npr

import scipy.stats as scits
import pandas as pd
import networkx as nx

import pysoft

import parser, graph


class GraphGenerator(object):
    """ Returns different types of network configurations (i.e. topologies, etc)
    """
    @staticmethod
    def get_er_graph(node_num=20, edge_prob=0.3):
        return graph.Graph(nx.erdos_renyi_graph(node_num, edge_prob, directed=True))

    @staticmethod
    def get_random_graph(node_num=20, activating_edges=0, inhibiting_edges=50):
        """ Create random graph by subsetting complete graph
        """
        g = nx.complete_graph(node_num, create_using=nx.DiGraph())
        edges = random.sample(g.edges(), activating_edges + inhibiting_edges)
        a_edges = edges[:activating_edges]
        i_edges = edges[-inhibiting_edges:]

        amat = np.zeros((node_num, node_num))
        for source, sink in a_edges: amat[source, sink] = 1
        for source, sink in i_edges: amat[source, sink] = -1

        gg = graph.Graph(nx.from_numpy_matrix(abs(amat), create_using=nx.DiGraph()))
        gg.aug_adja_m = amat

        return gg

    @staticmethod
    def get_regulatory_graph(file):
        return graph.Graph(parser.generate_tf_gene_regulation(file), largest=True)

    @staticmethod
    def get_minimal_graph():
        g = nx.DiGraph()

        g.add_edge(0,2)
        g.add_edge(1,0)
        g.add_edge(1,2)
        g.add_edge(2,1)

        return graph.Graph(g)

class StatsHandler(object):
    @staticmethod
    def correlate(x, y, compute_bands=False):
        """ Computes Pearson coefficient of x, y and compares it to the correlation of shuffled forms of x, y
        """
        (corr, p_val) = scits.pearsonr(x, y) # correlation r [-1,1], prob. that null-hypothesis (i.e. x,y uncorrelated) holds [0,1]

        if not compute_bands:
            return corr, p_val

        xs = np.copy(x)
        ys = np.copy(y)

        rs = []
        for i in range(3333):
            npr.shuffle(xs)
            npr.shuffle(ys)

            (r, p) = scits.pearsonr(xs, ys)
            rs.append(r)

        mi, ma = min(rs), max(rs)
        return corr, p_val, mi, ma

class DataHandler(object):
    backup_dir = 'conc_baks'

    @staticmethod
    def load_concentrations(graph, file, conc_range=[0]):
        """ Loads concentrations for given graph from given file and caches results for later reuse
        """
        bak_fname = os.path.join(DataHandler.backup_dir, 'conc_%s.bak' % os.path.basename(file))

        if os.path.isfile('%s.npy' % bak_fname):
            print('Recovering data from', bak_fname)
            foo = np.load('%s.npy' % bak_fname).item()
        else:
            print('Parsing data file', file)
            data = parser.parse_concentration(
                file,
                conc_range
            )

            concentrations = [t[1] for t in sorted(data.items(), key=operator.itemgetter(0))]
            if concentrations is None:
                return None
            try:
                concentrations = np.array(concentrations) / np.linalg.norm(concentrations)
            except ValueError:
                return None

            names = list(graph)
            #matched = set(names).intersection(set(data.keys()))
            no_match = set(names).difference(set(data.keys()))

            print('> coverage:', round(1 - len(no_match)/len(names), 3))

            foo = {
                'concentrations': concentrations,
                'map': data
            }

            # save for faster reuse
            if not os.path.exists(DataHandler.backup_dir):
                os.makedirs(DataHandler.backup_dir)
            np.save(bak_fname, foo)

        return foo['concentrations'], foo['map']

    @staticmethod
    def load_averaged_concentrations(graph, directory, conc_range=[0], cache=False):
        """ Loads concentration files in given directory and averages them
            In order to account for genes which appear in the graph but not in any dataset, this function will also return a vector of indices of the genes of the graph which were found in at least on dataset
        """
        # gather all data
        data = []
        genes = []
        for file in os.listdir(directory):
            if not file.endswith('.soft'): continue

            f = os.path.join(directory, file)
            c, m = DataHandler.load_concentrations(graph, f, conc_range)

            data.append(m)
            genes.append(set(m.keys()))

        # extract genes which are common to all data sets
        common_genes = sorted(set.intersection(*genes)) # genes which appear in all datasets
        all_genes = sorted(set.union(*genes).intersection(set(graph))) # genes which appear in at least one dataset
        print('-> Found', len(common_genes), 'common genes (%i covered in total)' % len(all_genes))

        # accumulate data
        res = []
        for gene in all_genes:
            gene_concs = []
            for exp in data:
                if gene in exp:
                    gene_concs.append(exp[gene])

            if cache:
                with open('RL_av_data.csv', 'a') as fd:
                    writer = csv.writer(fd)
                    writer.writerow(gene_concs)

            res.append(np.mean(gene_concs))

        # find used genes
        used_gene_indices = []
        for i, gene in enumerate(graph):
            if gene in all_genes:
                used_gene_indices.append(i)

        return np.array(res), used_gene_indices

class CacheHandler(object):
    cache_directory = 'plot_data'

    @staticmethod
    def store_plot_data(title, func, *args, model=None): #x_data, y_data, title, x_label, y_label):
        """ Store plot data and return sanitized data dict
        """
        if len(args) == 2:
            """ args = (x_label, data)
            """
            dic = {
                'data': args[1],
                'title': title,
                'x_label': args[0],
            }
        elif len(args) == 3:
            """ args = (x_label, y_label, data)
            """
            dic = {
                'data': args[2],
                'title': title,
                'x_label': args[0],
                'y_label': args[1]
            }
        elif len(args) == 4:
            """ args = (x_label, x_data, y_label, y_data)
            """
            dic = {
                'x_data': args[1],
                'y_data': args[3],
                'title': title,
                'x_label': args[0],
                'y_label': args[2]
            }

        model_id = 'raw'
        dic['info'] = {}
        if not model is None:
            dic['info'] = model.info
            model_id = model.hash()
        dic['info']['function'] = func.__name__

        if not os.path.exists(CacheHandler.cache_directory):
            os.makedirs(CacheHandler.cache_directory)
        CacheHandler.dump('%s__%s.dat' % (os.path.join(CacheHandler.cache_directory, clean_string(title)), model_id), dic)

        return dic

    @staticmethod
    def dump(fname, dic):
        """ Save dictionary to file (convert numpy arrays to python lists)
        """
        def clean(foo):
            t = type(foo)

            if t == type({}):
                return {k: clean(v) for (k,v) in foo.items()}
            elif t == type([]):
                return [clean(e) for e in foo]
            elif t == type(np.array([])):
                return foo.tolist()
            elif t == type(range(0)):
                return list(foo)
            return foo

        json.dump(clean(dic), open(fname, 'w'))

    @staticmethod
    def load(fname):
        return json.load(open(fname, 'r'))

class GDSHandler(object):
    """ Handles directories containing many GDS files
    """
    def __init__(self, dirname):
        self.dir = dirname

        self.common_genes = None

    def process_directory(self):
        """ Scan directory for SOFT files and extract gene concentration of genes which appear in all datasets
        """
        experiments = []
        genes = []
        for root, dirs, files in os.walk(self.dir):
            for fname in sorted(files):
                data = {}
                genes.append(set())

                soft = pysoft.SOFTFile(os.path.join(root, fname))

                for row in soft.data:
                    # keep all encountered genes
                    gene = row['IDENTIFIER'].lower()
                    genes[-1].add(gene)

                    # extract gene concentrations
                    conc = row[2]
                    if conc == 'null': conc = row[3]
                    if conc == 'null': conc = 0 # what to do now?
                    data[gene] = conc

                experiments.append(data)

        self.common_genes = set.intersection(*genes)

        # only extract common gene concentrations
        result = []
        for exp in experiments:
            tmp = {k: exp[k] for k in self.common_genes}
            result.append(tmp)

        return result


def clean_string(s):
    """ Make string useable as filename
    """
    return s.replace(' ', '_')

def df(**kwargs):
    """ Convert given args to proper dataframe
    """
    dic = {k: pd.Series(v, dtype='category') for (k,v) in kwargs.items()} # TODO: handle category properly
    return pd.DataFrame(dic)

def md5(s):
    m = hashlib.md5()
    m.update(s.encode(encoding='utf-8'))
    return m.hexdigest()
