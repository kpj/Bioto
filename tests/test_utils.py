from unittest import TestCase

import os.path, shutil

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

import networkx as nx

import utils


class TestMethods(TestCase):
    def test_string_sanitizer(self):
        self.assertEqual(utils.clean_string('filename with spaces.txt'), 'filename_with_spaces.txt')

    def test_md5_hasher(self):
        self.assertEqual(utils.md5('test123'), 'cc03e747a6afbbcbf8be7668acfebee5')

    def test_soft_detection(self):
        self.assertTrue(utils.is_soft_file('foo.soft'))
        self.assertTrue(utils.is_soft_file('foo.soft.gz'))
        self.assertFalse(utils.is_soft_file('foo.txt'))
        self.assertFalse(utils.is_soft_file('foo.txt.gz'))

class TestGraphGenerators(TestCase):
    def test_random_graph(self):
        graph = utils.GraphGenerator.get_random_graph(42, 30, 50)

        self.assertEqual(len(graph), 42)
        self.assertIsInstance(graph.graph, nx.DiGraph)

        pos_m = graph.aug_adja_m.copy()
        pos_m[pos_m != 1] = 0
        self.assertEqual(np.sum(pos_m), 30)

        neg_m = graph.aug_adja_m.copy()
        neg_m[neg_m != -1] = 0
        self.assertEqual(np.sum(neg_m), -50)

    def test_er_graph(self):
        graph = utils.GraphGenerator.get_er_graph(100, 0.66)

        self.assertEqual(len(graph), 100)
        self.assertIsInstance(graph.graph, nx.DiGraph)

class TestGDSHandler(TestCase):
    def setUp(self):
        self.gdsh = utils.GDSHandler('tests/data')

    def test_all_genes(self):
        res = self.gdsh.process_directory()

        self.assertEqual(len(res), 3)
        self.assertIn({'aaea': 42., 'aaeb': 2., 'zuzu': 23.}, res)
        self.assertIn({'aaea': 1337., 'aaeb': 4.}, res)
        self.assertIn({'aaea': 23., 'aaeb': 6.}, res)

    def test_common_genes(self):
        res = self.gdsh.process_directory(only_common_genes=True)

        self.assertEqual(len(res), 3)
        self.assertIn({'aaea': 42., 'aaeb': 2.}, res)
        self.assertIn({'aaea': 1337., 'aaeb': 4.}, res)
        self.assertIn({'aaea': 23., 'aaeb': 6.}, res)

class TestDataHandler(TestCase):
    def setUp(self):
        utils.DataHandler.backup_dir = 'conc_bak_test'

        network_file = 'tests/data/simple_network.txt'
        self.graph = utils.GraphGenerator.get_regulatory_graph(network_file)

    def tearDown(self):
        shutil.rmtree(utils.DataHandler.backup_dir)

    def test_single_file(self):
        conc_file = 'tests/data/foo.soft'
        concs, data = utils.DataHandler.load_concentrations(self.graph, conc_file)

        self.assertTrue(os.path.isdir(utils.DataHandler.backup_dir))
        self.assertTrue(os.path.isfile(os.path.join(utils.DataHandler.backup_dir, 'conc_foo.soft.bak.npy')))

        conc_vec = [42., 2., 23.]
        self.assertEqual(len(concs), 3)
        npt.assert_allclose(concs, conc_vec / npl.norm(conc_vec))

        self.assertEqual(len(data), 3)
        self.assertEqual(data['aaea'], 42.)
        self.assertEqual(data['aaeb'], 2.)
        self.assertEqual(data['zuzu'], 23.)

    def test_file_averaging(self):
        conc_dir = 'tests/data/'
        concs, used_gene_indices = utils.DataHandler.load_averaged_concentrations(self.graph, conc_dir)

        self.assertEqual(len(concs), 3)
        self.assertEqual(len(used_gene_indices), 3)

        npt.assert_allclose(concs, [467.3333333333333, 4., 23])
        npt.assert_allclose(used_gene_indices, [0, 1, 2])
