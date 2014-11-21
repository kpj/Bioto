import sys, os

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt
import numpy.random as npr

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

    def visualize(self, file, use_R=True):
        """ Visualize current graph and saves resulting image to specified file
        """
        if use_R:
            self.dump_adjacency_matrix('adja_m.txt')
            self.dump_node_names('node_names.txt')

            os.system('Rscript network_plotter.R "%s"' % file)
        else:
            pos = nx.random_layout(self.graph.graph)
            nx.draw(
                self.graph.graph, pos,
                with_labels=True,
                linewidths=0,
                width=0.1
            )
            plt.savefig(file, dpi=150)
            plt.close()

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

    def apply_power_iteration(self, eival, precision=1e-20, maxruns=1000):
        """ Alternative algorithm to find eigenvector of largest eigenvalue
            Useful on sparse matrices, but may converge slowly.
            As the norm converges to the eigenvalue of greatest magnitude, we stop when we're close enough
        """
        b = npr.sample(self.graph.adja_m.shape[0])

        while True:
            step = self.graph.adja_m.dot(b)
            norm = npl.norm(step)

            b = step / norm
            b = b.tolist()[0]

            if abs(norm - eival) < precision:
                break

            maxruns -= 1
            if maxruns == 0:
                print('Power Iteration Method did not converge, aborting...')
                break

        return np.array(b)

    def get_perron_frobenius(self, test_significance=True):
        """ Returns characteristic (normalized) Perron-Frobenius eigenvector
        """
        vals, vecs = npl.eig(self.graph.adja_m) # returns already normalized eigenvectors
        max_eigenvalue_index = np.argmax(np.real(vals))
        perron_frobenius = np.array(np.transpose(np.real(vecs[:, max_eigenvalue_index])).tolist()[0])

        if all(i <= 0 for i in perron_frobenius):
            perron_frobenius *= -1
        elif any(i < 0 for i in perron_frobenius):
            print("Error, pf-eigenvector is malformed")

        # check significance of result
        if test_significance:
            eival = vals[max_eigenvalue_index]
            if not self.test_significance(eival, perron_frobenius):
                print("pf-ev not sigificant, trying power iteration method")
                perron_frobenius = self.apply_power_iteration(eival)

            if not self.test_significance(eival, perron_frobenius):
                print("It didn't help, sorry")

        return perron_frobenius

    def test_significance(self, val, vec):
        """ Tests significance of eigenvaule/vector pair by checking Av=lv
            Return true if equality (approximatelly) holds
        """
        av = self.graph.adja_m.dot(vec).tolist()[0]
        lv = val * vec

        try:
            for i, j in zip(av, lv):
                npt.assert_approx_equal(i, np.real(j))
        except AssertionError:
            return False
        return True

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
