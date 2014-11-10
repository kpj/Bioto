import sys, os

import numpy as np
import numpy.linalg as npl

import sympy as sp

import matplotlib.pyplot as plt

import networkx as nx

import utils


class IOComponent(object):
    """ IO component for graph
    """

    def __init__(self, graph):
        self.graph = graph

    def dump_adjacency_matrix(self, file):
        """ Dumps adjacency matrix into specified file
        """
        np.savetxt(file, self.graph.adja_m)

    def dump_node_names(self, file):
        """ Dumps node names into specified file
        """
        with open(file, 'w') as fd:
            for node in self.graph:
                fd.write('%s\n' % node)

    def visualize(self, file):
        """ Visualize current graph and saves resulting image to specified file
        """
        self.dump_adjacency_matrix('adja_m.txt')
        self.dump_node_names('node_names.txt')

        os.system('Rscript network_plotter.R "%s"' % file)

    def load_concentrations(self, file):
        """ Delegates to utils.DataHandler
        """
        return utils.DataHandler.load_concentrations(self.graph, file)

    def load_averaged_concentrations(self, directory):
        """ Delegates to utils.DataHandler
        """
        return utils.DataHandler.load_averaged_concentrations(self.graph, directory)

class DynamicalSystem(object):
    def __init__(self, graph):
        self.graph = graph
        self.used_model = None

    def simulate(self, Model, runs=10):
        """ Simulates network evolution by using given model
            Model is passed as class, not as object
        """
        self.used_model = Model

        model = Model(self.graph)
        res = model.generate(runs)

        return res

class Math(object):
    def __init__(self, graph):
        self.graph = graph

    def get_perron_frobenius(self):
        """ Returns characteristic (normalized) Perron-Frobenius eigenvector
        """
        val, vec = npl.eig(self.graph.adja_m) # returns already normalized eigenvectors
        max_eigenvalue_index = np.argmax(np.real(val))
        perron_frobenius = np.array(np.transpose(np.real(vec[:, max_eigenvalue_index])).tolist()[0])

        #from sympy import pprint
        #import scipy.io
        #mat = sp.Matrix(self.graph.adja_m.tolist())
        #scipy.io.savemat('test.mat', dict(x=self.graph.adja_m))
        #eigs = mat.eigenvects()
        #perron_frobenius = np.array(max(eigs, key=lambda e: e[0])[2][0]).T[0]

        if all(i <= 0 for i in perron_frobenius):
            print("Rescaled pf-eigenvector by -1")
            perron_frobenius *= -1
        elif any(i < 0 for i in perron_frobenius):
            print("Error, pf-eigenvector is malformed")
            print(perron_frobenius)
            #sys.exit(1)
            return None

        return perron_frobenius

    def get_pagerank(self):
        """ Computes normalized page rank of current graph
        """
        pagerank = np.array(nx.pagerank(self.graph.graph)).tolist()

        vals = [v for v in pagerank.values()]
        vals /= npl.norm(vals)

        return vals

    def get_degree_distribution(self):
        """ Computes normalized degree distribution of current graph
        """
        deg_di = nx.degree(self.graph.graph).values()
        max_deg = max(deg_di)

        vals = [d/max_deg for d in deg_di]
        vals /= npl.norm(vals)

        return vals

class Graph(object):
    """ Central entity to conduct experiments/analyses on a given network
    """

    def __init__(self, graph, largest=False):
        """ Only considers largest weakly connected component by default
        """
        self.graph = graph
        if largest:
            self.graph = max(nx.weakly_connected_component_subgraphs(self.graph), key=len)

        self.adja_m = nx.to_numpy_matrix(self.graph)

        self.io = IOComponent(self)
        self.system = DynamicalSystem(self)
        self.math = Math(self)

    def __len__(self):
        """ Returns number of nodes in wrapped graph
        """
        return nx.number_of_nodes(self.graph)

    def __iter__(self):
        """ Returns node names in lowercase
        """
        for e in [str(n).lower() for n in nx.nodes_iter(self.graph)]:
            yield e
