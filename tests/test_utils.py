from unittest import TestCase

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

import networkx as nx

import utils


class UtilsTester(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_methods(self):
        self.assertEqual(utils.clean_string('filename with spaces.txt'), 'filename_with_spaces.txt')
        self.assertEqual(utils.md5('test123'), 'cc03e747a6afbbcbf8be7668acfebee5')

    def test_graph_generators(self):
        graph = utils.GraphGenerator.get_random_graph(42, 30, 50)

        self.assertEqual(len(graph), 42)

        pos_m = graph.aug_adja_m.copy()
        pos_m[pos_m != 1] = 0
        self.assertEqual(np.sum(pos_m), 30)

        neg_m = graph.aug_adja_m.copy()
        neg_m[neg_m != -1] = 0
        self.assertEqual(np.sum(neg_m), -50)

    def test_gdshandler(self):
        gdsh = utils.GDSHandler('tests/data')
        res = gdsh.process_directory()

        self.assertEqual(len(res), 3)
        self.assertIn({'aaea': '42', 'aaeb': '2'}, res)
        self.assertIn({'aaea': '1337', 'aaeb': '4'}, res)
        self.assertIn({'aaea': '23', 'aaeb': '6'}, res)
