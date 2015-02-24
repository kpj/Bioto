from unittest import TestCase

import numpy as np
import numpy.testing as npt

import file_parser


class TestParser(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_concentration_parser_single(self):
        gds_file = 'tests/data/foo.soft'

        data = file_parser.parse_concentration(gds_file)
        self.assertEqual(len(data), 3)
        self.assertEqual(data['aaea'], 42.)
        self.assertEqual(data['aaeb'], 2.)
        self.assertEqual(data['zuzu'], 23.)

        data = file_parser.parse_concentration(gds_file, conc_range=[1])
        self.assertEqual(len(data), 3)
        self.assertEqual(data['aaea'], 43.)
        self.assertEqual(data['aaeb'], 3.)
        self.assertEqual(data['zuzu'], 24.)

    def test_concentration_parser_multiple(self):
        gds_file = 'tests/data/foo.soft'
        data = file_parser.parse_concentration(gds_file, conc_range=[0, 1])

        self.assertEqual(len(data), 3)
        self.assertEqual(data['aaea'], [42., 43.])
        self.assertEqual(data['aaeb'], [2., 3.])
        self.assertEqual(data['zuzu'], [23., 24.])

    def test_regulation_parser(self):
        network_file = 'tests/data/simple_network.txt'
        data = file_parser.parse_regulation_file(network_file)

        self.assertEqual(len(data), 3)
        self.assertEqual(data['aaea'], [('aaeb', '+'), ('zuzu', '+')])
        self.assertEqual(data['aaeb'], [('aaea', '-')])
        self.assertEqual(data['zuzu'], [('zuzu', '-')])

    def test_trn_generation(self):
        network_file = 'tests/data/simple_network.txt'
        graph = file_parser.generate_tf_gene_regulation(network_file)

        self.assertEqual(len(graph), 3)
        self.assertIn('aaea', graph)
        self.assertIn('aaeb', graph)
        self.assertIn('zuzu', graph)

        self.assertEqual(len(graph.edges()), 4)
        self.assertIn(('aaea', 'aaeb'), graph.edges())
        self.assertIn(('aaea', 'zuzu'), graph.edges())
        self.assertIn(('aaeb', 'aaea'), graph.edges())
        self.assertIn(('zuzu', 'zuzu'), graph.edges())

    def test_augmented_adjacency_matrix_generation(self):
        network_file = 'tests/data/simple_network.txt'
        aug_adja = file_parser.get_advanced_adjacency_matrix(network_file)

        self.assertEqual(aug_adja.shape, (3, 3))
        npt.assert_allclose(aug_adja, np.array([[0,1,1], [-1,0,0], [0,0,-1]]))
