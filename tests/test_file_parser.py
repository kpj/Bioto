from unittest import TestCase

import numpy as np
import numpy.testing as npt

import networkx as nx

import file_parser


class TestParser(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_concentration_parser_single(self):
        gds_file = 'tests/data/foo.soft'

        data = file_parser.parse_concentration(gds_file)
        self.assertEqual(len(data), 4)
        self.assertEqual(data['aaea'], 42.)
        self.assertEqual(data['aaeb'], 2.)
        self.assertEqual(data['haba'], 1.)
        self.assertEqual(data['zuzu'], 23.)

        data = file_parser.parse_concentration(gds_file, conc_range=[1])
        self.assertEqual(len(data), 4)
        self.assertEqual(data['aaea'], 43.)
        self.assertEqual(data['aaeb'], 3.)
        self.assertEqual(data['haba'], 1.)
        self.assertEqual(data['zuzu'], 24.)

    def test_concentration_parser_multiple(self):
        gds_file = 'tests/data/foo.soft'
        data = file_parser.parse_concentration(gds_file, conc_range=[0, 1])

        self.assertEqual(len(data), 3)
        self.assertEqual(data['aaea'], [42., 43.])
        self.assertEqual(data['aaeb'], [2., 3.])
        self.assertEqual(data['zuzu'], [23., 24.])

    def test_regulation_parser(self):
        network_file = 'tests/data/trn_network.txt'
        data = file_parser.parse_regulation_file(network_file)

        self.assertEqual(len(data), 3)
        self.assertEqual(data['aaea'], [('aaeb', '+'), ('zuzu', '+')])
        self.assertEqual(data['aaeb'], [('aaea', '-')])
        self.assertEqual(data['zuzu'], [('zuzu', '-'), ('aaeb', '+-')])

    def test_trn_generation(self):
        network_file = 'tests/data/trn_network.txt'
        graph = file_parser.generate_tf_gene_regulation(network_file)

        self.assertIsInstance(graph, nx.DiGraph)

        self.assertEqual(len(graph), 3)
        self.assertIn('aaea', graph)
        self.assertIn('aaeb', graph)
        self.assertIn('zuzu', graph)

        self.assertEqual(len(graph.edges()), 5)
        self.assertIn(('aaea', 'aaeb'), graph.edges())
        self.assertIn(('aaea', 'zuzu'), graph.edges())
        self.assertIn(('aaeb', 'aaea'), graph.edges())
        self.assertIn(('zuzu', 'zuzu'), graph.edges())
        self.assertIn(('zuzu', 'aaeb'), graph.edges())

    def test_gene_proximity_parser(self):
        proxi_file = 'tests/data/gene_proximity_network.txt'
        data, max_right = file_parser.parse_gene_proximity_file(proxi_file)

        self.assertEqual(max_right, 70)
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0], {
            'name': 'yaaa',
            'left': 1,
            'right': 20
        })
        self.assertEqual(data[1], {
            'name': 'cydc',
            'left': 25,
            'right': 30
        })
        self.assertEqual(data[2], {
            'name': 'mepm',
            'left': 50,
            'right': 63
        })
        self.assertEqual(data[3], {
            'name': 'fnge',
            'left': 65,
            'right': 70
        })

    def test_gene_proximity_network_generation(self):
        proxi_file = 'tests/data/gene_proximity_network.txt'
        graph = file_parser.generate_gene_proximity_network(proxi_file, 10)

        self.assertIsInstance(graph, nx.MultiDiGraph)

        self.assertEqual(len(graph.nodes()), 4)
        self.assertEqual(set(graph.nodes()), set(['cydc', 'fnge', 'mepm', 'yaaa']))
        self.assertEqual(set(graph.edges()), set([('yaaa', 'cydc'), ('cydc', 'yaaa'), ('yaaa', 'mepm'), ('mepm', 'yaaa'), ('fnge', 'mepm'), ('mepm', 'fnge'), ('fnge', 'yaaa'), ('yaaa', 'fnge')]))

    def test_augmented_adjacency_matrix_generation(self):
        network_file = 'tests/data/trn_network.txt'
        aug_adja = file_parser.get_advanced_adjacency_matrix(network_file)

        self.assertEqual(aug_adja.shape, (3, 3))
        npt.assert_allclose(aug_adja, np.array([[0,1,1], [-1,0,0], [0,0,-1]]))
