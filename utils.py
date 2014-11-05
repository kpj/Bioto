import os, os.path

import numpy as np
import numpy.random as npr

import scipy.stats as scits

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

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

class StatsHandler(object):
    @staticmethod
    def correlate(x, y):
        """ Computes Pearson coefficient of x, y and compares it to the correlation of shuffled forms of x, y
        """
        (corr, null_hypo) = scits.pearsonr(x, y) # correlation r [-1,1], prob. that null-hypothesis (i.e. x,y uncorrelated) holds [0,1]

        xs = np.copy(x)
        ys = np.copy(y)

        rs = []
        for i in range(3333):
            npr.shuffle(xs)
            npr.shuffle(ys)

            (r, p) = scits.pearsonr(xs, ys)
            rs.append(r)

        mi, ma = min(rs), max(rs)
        return corr, mi, ma

class DataHandler(object):
    backup_dir = 'conc_baks'

    @staticmethod
    def load_concentrations(graph, file):
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
                file
            )
            concentrations = np.array(concentrations) / np.linalg.norm(concentrations)

            print('> coverage:', round(1 - len(fail)/len(names), 3))

            # save for faster reuse
            if not os.path.exists(DataHandler.backup_dir):
                os.makedirs(DataHandler.backup_dir)
            np.save(bak_fname, concentrations)

        return concentrations

    @staticmethod
    def load_averaged_concentrations(graph, directory):
        """ Loads concentration files in given directory and averages them
        """
        concs = []
        for file in os.listdir(directory):
            if not file.endswith('.soft'): continue

            f = os.path.join(directory, file)
            concs.append(DataHandler.load_concentrations(graph, f))

        res = []
        for col in np.array(concs).T:
            res.append(sum(col)/len(col))

        return np.array(res)

class Plotter(object):
    plot_save_directory = 'plot_dir'

    @staticmethod
    def present_graph(data, perron_frobenius, page_rank, degree_distribution):
        """ Shows a nice representation of the graphs features after evolution
        """
        info = [
            {
                'data': data[::-1],
                'title': 'Excitation Development via Adjacency-Matrix Multiplication',
                'rel_height': 6
            },
            {
                'data': np.array([perron_frobenius]),
                'title': 'Perron-Frobenius Eigenvector',
                'rel_height': 1
            },
            {
                'data': np.array([page_rank]),
                'title': 'Pagerank',
                'rel_height': 1
            },
            {
                'data': np.array([degree_distribution]),
                'title': 'Degree Distribution',
                'rel_height': 1
            }
        ]

        gs = gridspec.GridSpec(len(info), 1, height_ratios=[e['rel_height'] for e in info])
        for entry, g in zip(info, gs):
            ax = plt.subplot(g)

            ax.pcolor(entry['data'], cmap=cm.gray, vmin=-0.1, vmax=1)

            ax.set_title(entry['title'])
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        fig = plt.gcf()
        plt.show()

        if not os.path.exists(Plotter.plot_save_directory):
            os.makedirs(Plotter.plot_save_directory)
        fig.savefig(os.path.join(Plotter.plot_save_directory, 'overview.png'), dpi=150)

    @staticmethod
    def plot_loglog(x, y, title, xlabel, ylabel):
        """ Creates loglog plot of given data and removes 0-pairs beforehand
        """
        ax = plt.gca()

        xs = []
        ys = []
        for i, j in zip(x, y):
            # remove 0-pairs
            if not (i == 0 or j == 0):
                xs.append(i)
                ys.append(j)

        ax.loglog(
            xs, ys,
            linestyle='None',
            marker='.', markeredgecolor='blue'
        )

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        fig = plt.gcf()
        plt.show()

        if not os.path.exists(Plotter.plot_save_directory):
            os.makedirs(Plotter.plot_save_directory)
        fig.savefig(os.path.join(Plotter.plot_save_directory, '%s.png' % title.replace(' ', '_')), dpi=150)
