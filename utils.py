import sys
import os, os.path

import numpy as np
import numpy.random as npr
import numpy.linalg as npl

import scipy.stats as scits

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

import networkx as nx

import parser


class GraphGenerator(object):
    """ Returns different types of network configurations (i.e. topologies, etc)
    """
    @staticmethod
    def get_random_graph(node_num=20, edge_prob=0.6):
        return nx.gnp_random_graph(node_num, edge_prob)

    @staticmethod
    def get_regulatory_graph(file):
        return parser.generate_tf_gene_regulation(file)

class GraphHandler(object):
    """ Central entity to conduct experiments/analyses on a given network
    """
    def __init__(self, graph):
        self.graph = graph
        self.adja_m = nx.to_numpy_matrix(self.graph)

    def __len__(self):
        """ Returns number of nodes in wrapped graph
        """
        return nx.number_of_nodes(self.graph)

    def dump_adjacency_matrix(self, file):
        """ Dumps adjacency matrix into specified file
        """
        np.savetxt(file, self.adja_m)

    def dump_node_names(self, file):
        """ Dumps node names into specified file
        """
        for n in self.get_node_names():
            fd.write('%s\n' % n)

    def get_node_names(self):
        """ Returns node names in lowercase
        """
        return [n.lower() for n in nx.nodes_iter(self.graph)]

    def visualize(self, file):
        """ Visualize current graph and saves resulting image to specified file
        """
        pos = nx.random_layout(self.graph)
        nx.draw(
            self.graph, pos,
            with_labels=True,
            linewidths=0,
            width=0.1
        )
        plt.savefig(file, dpi=150)

    def simulate(self, runs=11, initial=None):
        """ Simulates network evolution by adjacency matrix multiplication
        """
        if not initial:
            initial = np.array([npr.random() for i in range(self.graph.number_of_nodes())])

        data = [initial]
        for i in range(runs):
            cur = [self.adja_m.dot(data[-1])[0, i] for i in range(self.graph.number_of_nodes())]
            cur /= npl.norm(cur)

            data = np.vstack((data, cur))

        return data

    def get_perron_frobenius(self):
        """ Returns characteristic (normalized) Perron-Frobenius eigenvector
        """
        val, vec = npl.eig(self.adja_m) # returns already normalized eigenvectors
        max_eigenvalue_index = np.argmax(np.real(val))
        perron_frobenius = np.array(np.transpose(np.real(vec[:, max_eigenvalue_index])).tolist()[0])

        #sorted_vals = sorted(np.real(val))
        #x = (sorted_vals[-1]-sorted_vals[-2])/sorted_vals[-1]
        #y = min(sum(perron_frobenius<0), sum(perron_frobenius>0)) # find number of mismatching entries after proper rescaling

        if all(i <= 0 for i in perron_frobenius):
            print("Rescaled pf-eigenvector by -1")
            perron_frobenius *= -1
        elif any(i < 0 for i in perron_frobenius):
            print("Error, pf-eigenvector is malformed")
            #print(perron_frobenius)
            sys.exit(1)
            return None

        return perron_frobenius

    def get_pagerank(self):
        """ Computes normalized page rank of current graph
        """
        pagerank = np.array(nx.pagerank(self.graph)).tolist()

        vals = [v for v in pagerank.values()]
        vals /= npl.norm(vals)

        return vals

    def get_degree_distribution(self):
        """ Computes normalized degree distribution of current graph
        """
        deg_di = nx.degree(self.graph).values()
        max_deg = max(deg_di)

        vals = [d/max_deg for d in deg_di]
        vals /= npl.norm(vals)

        return vals

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
    @staticmethod
    def load_concentrations(graph, file):
        """ Loads concentrations for given graph from given file and caches results for later reuse
        """
        bak_fname = 'conc_%s.bak' % os.path.basename(file)

        if os.path.isfile('%s.npy' % bak_fname):
            print('Recovering data from', bak_fname)
            concentrations = np.load('%s.npy' % bak_fname)
        else:
            print('Parsing data file', file)
            names = graph.get_node_names()
            concentrations, fail = parser.parse_concentration(
                names,
                file
            )
            concentrations = np.array(concentrations) / np.linalg.norm(concentrations)

            print('> coverage:', round(1 - len(fail)/len(names), 3))

            # save for faster reuse
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

        fig = plt.figure()
        gs = gridspec.GridSpec(len(info), 1, height_ratios=[e['rel_height'] for e in info])
        for entry, g in zip(info, gs):
            ax = plt.subplot(g)

            ax.pcolor(entry['data'], cmap=cm.gray, vmin=-0.1, vmax=1)

            ax.set_title(entry['title'])
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        plt.show()

    @staticmethod
    def plot_loglog(x, y, title, xlabel, ylabel):
        """ Creates loglog plot of given data and removes 0-pairs beforehand
        """
        fig = plt.figure()
        ax = plt.gca()

        xs = []
        ys = []
        for i, j in zip(x, y):
            # remove 0-pairs
            if not (i == 0 or j == 0):
                xs.append(i)
                ys.append(j)

        ax.loglog(xs, ys)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.show()
