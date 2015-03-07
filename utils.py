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

import file_parser, graph, errors


class GraphGenerator(object):
    """ Return different types of network configurations (i.e. topologies, etc)
    """
    @staticmethod
    def get_er_graph(node_num=20, edge_prob=0.3):
        """ Return simple ER-graph
        """
        return graph.Graph(nx.erdos_renyi_graph(node_num, edge_prob, directed=True))

    @staticmethod
    def get_random_graph(graphdef, activating_edges=0, inhibiting_edges=0):
        """ Create random graph by subsetting complete graph
        """
        if isinstance(graphdef, int):
            node_num = graphdef
            g = nx.complete_graph(graphdef, create_using=nx.DiGraph())
        elif isinstance(graphdef, graph.Graph):
            node_num = len(graphdef)
            g = graphdef.graph
        else:
            raise RuntimeError('Invalid graph definition given')

        edges = random.sample(g.edges(), activating_edges + inhibiting_edges)
        a_edges = edges[:activating_edges]
        i_edges = edges[-inhibiting_edges:] if inhibiting_edges > 0 else []

        amat = np.zeros((node_num, node_num))
        for source, sink in a_edges: amat[source, sink] = 1
        for source, sink in i_edges: amat[source, sink] = -1

        gg = graph.Graph(nx.from_numpy_matrix(abs(amat), create_using=nx.DiGraph()))
        gg.aug_adja_m = amat

        return gg

    @staticmethod
    def get_regulatory_graph(tf_reg_file, gene_proximity_file=None, base_window=50000, reduce_gpn=True):
        """ Return transcriptional regulatory network specified by given file
            If gene_proximity_file is given, TRN and GPN will be composed to a MultiDiGraph
        """
        trn = graph.Graph(file_parser.generate_tf_gene_regulation(tf_reg_file), largest=True)

        if not gene_proximity_file is None:
            gpn = GraphGenerator.get_gene_proximity_network(gene_proximity_file, base_window)
            if reduce_gpn: gpn.reduce_to(trn)

            trn += gpn

        return trn

    @staticmethod
    def get_gene_proximity_network(gene_prox_file, base_window):
        """ Return gene proximity network specified by given file
        """
        return graph.Graph(file_parser.generate_gene_proximity_network(gene_prox_file, base_window))

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
        """ Extract gene concentrations from file which also appear in graph.
            Also return vector of node indices used in concentration vector
        """
        bak_fname = os.path.join(DataHandler.backup_dir, 'conc_%s.bak' % os.path.basename(file))

        if os.path.isfile('%s.npy' % bak_fname):
            print('Recovering data from', bak_fname)
            foo = np.load('%s.npy' % bak_fname).item()
        else:
            print('Parsing data file', file)
            gdsh = GDSHandler(os.path.dirname(file))
            data = gdsh.parse_file(os.path.basename(file), conc_range)

            concentrations = []
            used_gene_indices = []
            for i, gene in enumerate(graph):
                if gene in data:
                    concentrations.append(data[gene])
                    used_gene_indices.append(i)

            #matched = set(graph).intersection(set(data.keys()))
            no_match = set(graph).difference(set(data.keys()))
            print('> coverage:', round(1 - len(no_match)/len(graph), 3))

            foo = {
                'concentrations': concentrations,
                'used_gene_indices': used_gene_indices
            }

            # cache for faster reuse
            if not os.path.exists(DataHandler.backup_dir):
                os.makedirs(DataHandler.backup_dir)
            np.save(bak_fname, foo)

        return foo['concentrations'], foo['used_gene_indices']

    @staticmethod
    def load_averaged_concentrations(graph, directory, conc_range=[0], cache_file=None):
        """ Loads concentration files in given directory and averages them
            In order to account for genes which appear in the graph but not in any dataset, this function will also return a vector of indices of the genes of the graph which were found in at least on dataset
        """
        # gather all data
        gdsh = GDSHandler(directory)
        experis = gdsh.process_directory()

        common_genes_in_graph = sorted(set(gdsh.common_genes).intersection(set(graph)))
        genes_in_graph = sorted(set(gdsh.all_genes).intersection(set(graph)))
        print('-> Found', len(common_genes_in_graph), 'common genes (%i covered in total)' % len(genes_in_graph))

        # accumulate data
        res = []
        for gene in genes_in_graph:
            gene_concs = []
            for exp in experis:
                if gene in exp:
                    gene_concs.append(exp[gene])

            if not cache_file is None:
                with open(cache_file, 'a') as fd:
                    writer = csv.writer(fd)
                    writer.writerow(gene_concs)

            res.append(np.mean(gene_concs))

        # find used genes
        used_gene_indices = []
        for i, gene in enumerate(graph):
            if gene in genes_in_graph:
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
    def dump(fname, obj):
        """ Save dictionary to file (convert numpy arrays to python lists)
        """
        def clean(foo):
            t = type(foo)

            if t == type({}):
                return {k: clean(v) for (k,v) in foo.items()}
            elif t == type([]):
                return [clean(e) for e in foo]
            elif t == type(np.array([])):
                return clean(foo.tolist())
            elif t == type(range(0)):
                return list(foo)
            return foo

        with open(fname, 'w') as fd:
            json.dump(clean(obj), fd)

    @staticmethod
    def load(fname):
        with open(fname, 'r') as fd:
            return json.load(fd)

class GDSHandler(object):
    """ Handles directories containing many GDS files
    """
    def __init__(self, dirname):
        self.dir = dirname

        self.all_genes = None # genes which appear in at least one dataset
        self.common_genes = None # genes which appear in all datasets

    def parse_file(self, fname, conc_range=[0]):
        """ Extract all (valid) gene concentrations from file
            fname is relative to dirname given in constructor
        """
        return file_parser.parse_concentration(os.path.join(self.dir, fname), conc_range)

    def process_directory(self, only_common_genes=False):
        """ Scan directory for SOFT files.
            Extract gene concentration of genes which appear in all datasets if requested
        """
        experiments = []
        genes = []
        for root, dirs, files in os.walk(self.dir):
            for fname in sorted(files):
                if not is_soft_file(fname): continue

                try:
                    data = self.parse_file(fname)
                except errors.InvalidGDSFormatError:
                    continue
                if len(data) == 0: continue

                experiments.append(data)

                genes.append(set())
                genes[-1] = set(data.keys())

        self.all_genes = sorted(set.union(*genes))
        self.common_genes = sorted(set.intersection(*genes))

        # extract gene concentrations
        result = []
        genes_to_extract = self.common_genes if only_common_genes else self.all_genes
        for exp in experiments:
            tmp = {k: exp[k] if k in exp else None for k in genes_to_extract}
            tmp = dict(filter(lambda x: not x[1] is None, tmp.items()))
            result.append(tmp)

        return result

class GDSFormatHandler(object):
    """ Handle all the different file formats GDS files may come in
    """
    def __init__(self, soft):
        self.soft = soft
        self.type = self.soft.header['dataset']['dataset_value_type']

    def get_data(self):
        for row in self.soft.data:
            yield self.parse_row(row)

    def parse_row(self, row):
        if self.type == 'log2 ratio':
            return row

        raise errors.InvalidGDSFormatError('Encountered invalid format "%s"' % self.type)


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

def is_soft_file(fname):
    return fname.endswith('.soft') or fname.endswith('.soft.gz')
