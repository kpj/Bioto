from unittest import TestCase

import numpy as np
import numpy.testing as npt

import networkx as nx

import graph


class GraphTester(TestCase):
    def setUp(self):
        self.raw_graph = nx.DiGraph()
        self.raw_graph.add_nodes_from([1, 2, 3, 4])
        self.raw_graph.add_edges_from([(1,2), (2,3), (3,2), (3,1)])
        self.graph = graph.Graph(self.raw_graph, largest=True)

    def tearDown(self):
        pass

    def test_initialization(self):
        self.assertEqual(len(self.graph), 3)
        self.assertEqual(list(self.graph), ['1', '2', '3'])
        npt.assert_array_equal(self.graph.adja_m, np.array([[0,1,0], [0,0,1], [1,1,0]]))


        # don't just take largest subcomponent
        whole_graph = graph.Graph(self.raw_graph)

        self.assertEqual(len(whole_graph), 4)
        self.assertEqual(list(whole_graph), ['1', '2', '3', '4'])
        npt.assert_array_equal(whole_graph.adja_m, np.array([[0,1,0,0], [0,0,1,0], [1,1,0,0], [0,0,0,0]]))
