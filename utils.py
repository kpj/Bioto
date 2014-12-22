import os, os.path
import json
import hashlib

import numpy as np
import numpy.random as npr

import scipy.stats as scits

import pandas as pd

import networkx as nx

import parser, graph


class GraphGenerator(object):
    """ Returns different types of network configurations (i.e. topologies, etc)
    """
    @staticmethod
    def get_random_graph(node_num=20, edge_prob=0.3):
        return graph.Graph(nx.erdos_renyi_graph(node_num, edge_prob, directed=True))

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
            concentrations = np.load('%s.npy' % bak_fname)
        else:
            print('Parsing data file', file)
            names = list(graph)
            concentrations, fail = parser.parse_concentration(
                names,
                file,
                conc_range
            )
            if concentrations is None:
                return None

            try:
                concentrations = np.array(concentrations) / np.linalg.norm(concentrations)
            except ValueError:
                return None

            print('> coverage:', round(1 - len(fail)/len(names), 3))

            # save for faster reuse
            if not os.path.exists(DataHandler.backup_dir):
                os.makedirs(DataHandler.backup_dir)
            np.save(bak_fname, concentrations)

        return concentrations

    @staticmethod
    def load_averaged_concentrations(graph, directory, conc_range=[0]):
        """ Loads concentration files in given directory and averages them
        """
        concs = []
        for file in os.listdir(directory):
            if not file.endswith('.soft'): continue

            f = os.path.join(directory, file)
            c = DataHandler.load_concentrations(graph, f, conc_range)

            if not c is None:
                concs.append(c)

        res = []
        for col in np.array(concs).T:
            res.append(sum(col)/len(col))

        return np.array(res)

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
            model_id = md5(repr(sorted(model.info.items())))
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


def clean_string(s):
    """ Make string useble as filename
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
