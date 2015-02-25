import sys, subprocess

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
        np.savetxt(file, self.graph.adja_m, fmt='%d')

    def dump_node_names(self, file):
        """ Dumps node names into specified file
        """
        with open(file, 'w') as fd:
            for node in self.graph:
                fd.write('%s\n' % node)

    def visualize(self, file, use_R=True, verbose=False):
        """ Visualize current graph and saves resulting image to specified file
        """
        if use_R:
            self.dump_adjacency_matrix('adja_m.txt')
            self.dump_node_names('node_names.txt')

            subprocess.check_call(['Rscript', 'network_plotter.R', file], stdout=None if verbose else subprocess.DEVNULL, stderr=None if verbose else subprocess.DEVNULL)
        else:
            pos = nx.random_layout(self.graph.graph)
            nx.draw(
                self.graph.graph, pos,
                with_labels=True,
                linewidths=1,
                width=0.1,
                node_color='#AAFFBB'
            )
            plt.savefig(file, dpi=150)
            plt.close()

    def load_concentrations(self, file, conc_range=[0]):
        """ Delegates to utils.DataHandler
        """
        return utils.DataHandler.load_concentrations(self.graph, file, conc_range)

    def load_averaged_concentrations(self, directory, conc_range=[0], cache_file=None):
        """ Delegates to utils.DataHandler
        """
        return utils.DataHandler.load_averaged_concentrations(self.graph, directory, conc_range, cache_file)

class DynamicalSystem(object):
    def __init__(self, graph):
        self.graph = graph
        self.used_model = None

    def simulate(self, Model=None, **kwargs):
        """ Simulates network evolution by using given model
            Model is passed as class, not as object
        """
        if Model is None:
            if self.used_model is None:
                raise RuntimeError('No model was assigned to this graph. Please do so.')
            else:
                print('Reusing previously selected model.')
                model = self.used_model
        else:
            model = Model(self.graph, **kwargs)
            self.used_model = model

        res = model.generate(**kwargs)
        return res

class Math(object):
    def __init__(self, graph):
        self.graph = graph

    def apply_power_iteration(self, eival, mat=None, precision=1e-20, maxruns=1000):
        """ Alternative algorithm to find eigenvector of largest eigenvalue
            Useful on sparse matrices, but may converge slowly.
            As the norm converges to the eigenvalue of greatest magnitude, we stop when we're close enough
        """
        if mat is None:
            mat = self.graph.adja_m

        b = npr.sample(mat.shape[0])

        while True:
            step = mat.dot(b)
            norm = npl.norm(step)

            b = step / norm

            if abs(norm - eival) < precision:
                break

            maxruns -= 1
            if maxruns == 0:
                raise RuntimeError('Power Iteration Method did not converge, aborting...')

        return np.array(b)

    def get_perron_frobenius(self, mat=None, test_significance=True, real_eva_only=False, rescale=True, remove_self_links=False):
        """ Returns characteristic (normalized) Perron-Frobenius eigenvector
        """
        if mat is None:
            mat = self.graph.adja_m

        mat = np.array(mat) # deal with all those formats which might arrive here

        if remove_self_links:
            np.fill_diagonal(mat, 0)

        vals, vecs = npl.eig(mat) # returns already normalized eigenvectors
        if real_eva_only:
            lv = -float('inf')
            for i, v in enumerate(vals):
                if np.isreal(v):
                    if v > lv:
                        max_eigenvalue_index = i
                        lv = v
        else:
            max_eigenvalue_index = np.argmax(np.real(vals))

        perron_frobenius = np.real(vecs[:, max_eigenvalue_index])
        perron_frobenius[perron_frobenius < 1e-13] = 0 # account for numeric instabilities

        if rescale:
            if all(i <= 0 for i in perron_frobenius):
                perron_frobenius *= -1
            elif any(i < 0 for i in perron_frobenius):
                print("Error, pf-eigenvector is malformed")

        # check significance of result
        if test_significance:
            eival = vals[max_eigenvalue_index]
            if not self.test_significance(eival, perron_frobenius, mat=mat):
                print("pf-ev not significant, trying averaged power iteration method")

                pfs = []
                for i in range(50):
                    pfs.append(self.apply_power_iteration(eival))
                perron_frobenius = sum(pfs)/len(pfs)

            if not self.test_significance(eival, perron_frobenius, mat=mat):
                print("It didn't help, sorry")

        return perron_frobenius

    def test_significance(self, val, vec, mat=None):
        """ Tests significance of eigenvaule/vector pair by checking Av=lv
            Return true if equality (approximatelly) holds
        """
        if mat is None:
            mat = self.graph.adja_m

        av = mat.dot(vec)
        lv = val * vec

        try:
            npt.assert_allclose(av, lv)
        except AssertionError:
            return False
        return True

    def get_pagerank(self, damping_factor=0.85):
        """ Computes normalized page rank of current graph
        """
        pagerank = np.array(nx.pagerank(self.graph.graph, alpha=damping_factor)).tolist()

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

        self.adja_m = np.array(nx.to_numpy_matrix(self.graph, nodelist=sorted(self.graph.nodes())), dtype=np.int8)
        self.aug_adja_m = None

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
        for e in sorted([str(n).lower() for n in nx.nodes_iter(self.graph)]):
            yield e
