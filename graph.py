import pickle, datetime
import os, os.path
import sys, subprocess

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt
import numpy.random as npr

import sympy as sp

import matplotlib.pyplot as plt

import networkx as nx

import utils, errors


class IOComponent(object):
    """ IO component for graph
    """
    DUMP_DIR = 'graph_dumps'

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

    def load_concentrations(self, fname, conc_range=None, **kwargs):
        """ Delegates to utils.DataHandler
        """
        if fname.endswith('.soft'):
            # is microarray data
            return utils.DataHandler.load_concentrations(self.graph, fname, conc_range, **kwargs)
        elif fname.endswith('.count'):
            return utils.DataHandler.load_rnaseq_data(self.graph, fname, **kwargs)
        else:
            print('Could not parse file "%s", unknown format' % fname)

    def load_averaged_concentrations(self, directory, conc_range=None, cache_file=None):
        """ Delegates to utils.DataHandler
        """
        return utils.DataHandler.load_averaged_concentrations(self.graph, directory, conc_range, cache_file)

    def dump(self, fname=None):
        """ Dump graph object to file
        """
        if not os.path.isdir(IOComponent.DUMP_DIR):
            os.mkdir(IOComponent.DUMP_DIR)

        if fname is None:
            fname = '%s.grph' % datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        with open(os.path.join(IOComponent.DUMP_DIR, fname), 'wb') as fd:
            pickle.dump(self.graph, fd)

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
                raise errors.PowerIterationError('Power Iteration Method did not converge, aborting...')

        return np.array(b)

    def get_perron_frobenius(self, mat=None, test_significance=True, real_eva_only=False, rescale=True, remove_self_links=False):
        """ Return characteristic (normalized) Perron-Frobenius eigenvector
        """
        if mat is None:
            mat = self.graph.adja_m

        mat = np.array(mat) # deal with all those formats which might arrive here

        if remove_self_links:
            np.fill_diagonal(mat, 0)

        vals, vecs = npl.eig(mat) # returns already normalized eigenvectors
        max_eigenvalue_index = utils.get_max_entry_index(vals, real_entries_only=real_eva_only)

        perron_frobenius = np.real(vecs[:, max_eigenvalue_index])
        perron_frobenius[abs(perron_frobenius) < 1e-13] = 0 # account for numeric instabilities

        if rescale:
            if all(i <= 0 for i in perron_frobenius):
                perron_frobenius *= -1
            elif any(i < 0 for i in perron_frobenius):
                raise errors.PFComputationError('pf-eigenvector is malformed')

        # check significance of result
        if test_significance:
            eival = vals[max_eigenvalue_index]
            if not self.test_significance(eival, perron_frobenius, mat=mat):
                pfs = []
                for i in range(50):
                    pfs.append(self.apply_power_iteration(eival))
                perron_frobenius = sum(pfs)/len(pfs)

                if not self.test_significance(eival, perron_frobenius, mat=mat):
                    raise errors.PFComputationError('pf-eigenvector is not significant')

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
        pagerank = np.array(nx.pagerank_numpy(self.graph.graph, alpha=damping_factor)).tolist()

        vals = list(pagerank.values())
        vals /= npl.norm(vals)

        return vals

    def get_degree_distribution(self, indegree=True):
        """ Computes normalized degree distribution of current graph
            The degree is either in- [default], out- or overall-degree
        """
        if indegree:
            degs = self.graph.graph.in_degree()
        elif indegree is None:
            degs = self.graph.graph.degree()
        else:
            degs = self.graph.graph.out_degree()

        degs = {str(k): v for k, v in degs.items()}
        deg_di = [degs[node] for node in self.graph]
        deg_di /= npl.norm(deg_di)

        return deg_di

class Graph(object):
    """ Central entity to conduct experiments/analyses on a given network
    """

    @staticmethod
    def from_file(fname):
        """ Generate graph from file
        """
        with open(fname, 'rb') as fd:
            tmp = pickle.load(fd)
            #if not isinstance(tmp, Graph):
            #    raise TypeError('Tried to load invalid graph file')

            return tmp

    def __init__(self, graph, largest=False):
        """ Only considers largest weakly connected component if needed
        """
        self.graph = graph
        if largest:
            self.graph = max(nx.weakly_connected_component_subgraphs(self.graph), key=len)

        self.io = IOComponent(self)
        self.system = DynamicalSystem(self)
        self.math = Math(self)

        self.setup()

    def setup(self):
        self.adja_m = np.array(nx.to_numpy_matrix(self.graph, nodelist=sorted(self.graph.nodes())), dtype=np.int8)
        self.aug_adja_m = None

    def reduce_to(self, graph):
        """ Reduce current graph to only possess the nodes of the given graph
        """
        own_nodes = set(self)
        his_nodes = set(graph)

        self.graph.remove_nodes_from(own_nodes.difference(his_nodes))
        self.setup()

    def get_components(self):
        """ Return list of weakly connected sub-components or list with only itself if graph is connected
            (weakly connected: connected after after raplcing all directed edges with undirected ones)
        """
        for subg in nx.weakly_connected_component_subgraphs(self.graph):
            yield Graph(subg)

    def __len__(self):
        """ Returns number of nodes in wrapped graph
        """
        return nx.number_of_nodes(self.graph)

    def __iter__(self):
        """ Returns node names in lowercase
        """
        for e in sorted([str(n).lower() for n in nx.nodes_iter(self.graph)]):
            yield e

    def __add__(self, right):
        """ Compose graphs
        """
        if not isinstance(right, Graph):
            raise TypeError('Can only compose two graph objects')

        my_graph = self.graph
        his_graph = right.graph

        if isinstance(my_graph, nx.DiGraph) and isinstance(his_graph, nx.MultiDiGraph):
            my_graph = nx.MultiDiGraph(my_graph)

        return Graph(nx.compose(my_graph, his_graph))

    def __lt__(self, other):
        return len(self) < len(other)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <graph file>' % sys.argv[0])
        sys.exit(1)

    graph = Graph.from_file(sys.argv[1])
