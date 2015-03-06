from unittest import TestCase

import os.path, shutil

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

import networkx as nx

import pysoft

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

    def test_data_frame_creator(self):
        res = utils.df(foo=[23,42], bar=['baz','qux'])

        self.assertEqual(len(res), 2)
        npt.assert_allclose(res['foo'], [23, 42])
        npt.assert_array_equal(res['bar'], ['baz', 'qux'])

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

    def test_trn_graph(self):
        trn_file = 'tests/data/trn_network.txt'
        gpn_file = 'tests/data/gene_proximity_network.txt'

        # without gpn is tested for file_parser, test TRN/GPN concatenation
        graph = utils.GraphGenerator.get_regulatory_graph(trn_file, gpn_file, base_window=10, reduce_gpn=False)

        self.assertIsInstance(graph.graph, nx.MultiDiGraph)
        self.assertEqual(len(graph), 7)
        self.assertEqual(list(graph), ['aaea', 'aaeb', 'cydc', 'fnge', 'mepm', 'yaaa', 'zuzu'])

        self.assertEqual(set(graph.graph.edges()), set([('yaaa', 'cydc'), ('cydc', 'yaaa'), ('yaaa', 'mepm'), ('mepm', 'yaaa'), ('fnge', 'mepm'), ('mepm', 'fnge'), ('fnge', 'yaaa'), ('yaaa', 'fnge'), ('aaea', 'aaeb'), ('aaea', 'zuzu'), ('aaeb', 'aaea'), ('zuzu', 'zuzu'), ('zuzu', 'aaeb')]))

class TestStatsHandler(TestCase):
    def test_pearson_correlation(self):
        v = list(range(0, 100))

        cp = utils.StatsHandler().correlate(v, v)
        self.assertEqual(cp, (1., 0.))

        cp = utils.StatsHandler().correlate(v, list(reversed(v)))
        self.assertEqual(cp, (-1., 0.))

        c, p, mi, ma = utils.StatsHandler().correlate(v, v, compute_bands=True)
        self.assertEqual((c, p), (1., 0.))
        self.assertTrue(mi > -1)
        self.assertTrue(ma < 1)

class TestCacheHandler(TestCase):
    def test_json_handling(self):
        tmp_file = 'tmp_testing.json'

        # general stuff
        struct = [{'3': [1,2,3]}]
        utils.CacheHandler.dump(tmp_file, struct)
        res = utils.CacheHandler.load(tmp_file)
        self.assertEqual(struct, res)

        # numpy array conversion
        np_struct = np.array([{'3': np.array([1,2,3])}])
        utils.CacheHandler.dump(tmp_file, np_struct)
        res = utils.CacheHandler.load(tmp_file)
        self.assertEqual(struct, res)

        # range conversion
        range_struct = [{'3': range(1,4)}]
        utils.CacheHandler.dump(tmp_file, range_struct)
        res = utils.CacheHandler.load(tmp_file)
        self.assertEqual(struct, res)

        os.remove(tmp_file)

class TestGDSHandler(TestCase):
    def setUp(self):
        self.gdsh = utils.GDSHandler('tests/data')

    def test_single_file(self):
        gds_file = 'foo.soft'
        res = self.gdsh.parse_file(gds_file)

        self.assertEqual(res, {
            'aaea': 42.,
            'aaeb': 2.,
            'haba': 1.,
            'zuzu': 23.
        })

    def test_all_genes(self):
        res = self.gdsh.process_directory()

        self.assertEqual(len(res), 3)
        self.assertIn({'aaea': 42., 'aaeb': 2., 'haba': 1., 'zuzu': 23.}, res)
        self.assertIn({'aaea': 1337., 'aaeb': 4.}, res)
        self.assertIn({'aaea': 23., 'aaeb': 6.}, res)

        self.assertEqual(len(self.gdsh.all_genes), 4)
        self.assertEqual(len(self.gdsh.common_genes), 2)

    def test_common_genes(self):
        res = self.gdsh.process_directory(only_common_genes=True)

        self.assertEqual(len(res), 3)
        self.assertIn({'aaea': 42., 'aaeb': 2.}, res)
        self.assertIn({'aaea': 1337., 'aaeb': 4.}, res)
        self.assertIn({'aaea': 23., 'aaeb': 6.}, res)

        self.assertEqual(len(self.gdsh.all_genes), 4)
        self.assertEqual(len(self.gdsh.common_genes), 2)

class TestDataHandler(TestCase):
    def setUp(self):
        utils.DataHandler.backup_dir = 'conc_bak_test'

        network_file = 'tests/data/trn_network.txt'
        self.graph = utils.GraphGenerator.get_regulatory_graph(network_file)

    def tearDown(self):
        try:
            shutil.rmtree(utils.DataHandler.backup_dir)
        except FileNotFoundError:
            pass

    def test_empty_file(self):
        conc_file = 'tests/data/qux.soft'
        concs, ugi = utils.DataHandler.load_concentrations(self.graph, conc_file)

        self.assertEqual(len(concs), 0)
        self.assertEqual(len(ugi), 0)

    def test_single_file(self):
        conc_file = 'tests/data/foo.soft'
        concs, ugi = utils.DataHandler.load_concentrations(self.graph, conc_file)

        self.assertTrue(os.path.isdir(utils.DataHandler.backup_dir))
        self.assertTrue(os.path.isfile(os.path.join(utils.DataHandler.backup_dir, 'conc_foo.soft.bak.npy')))

        npt.assert_allclose(concs, [42., 2., 23.])
        npt.assert_allclose(ugi, [0, 1, 2])

    def test_file_averaging(self):
        conc_dir = 'tests/data/'
        cache_file = 'RL_av_data_testing.csv'
        concs, used_gene_indices = utils.DataHandler.load_averaged_concentrations(self.graph, conc_dir, cache_file=cache_file)

        self.assertEqual(len(concs), 3)
        self.assertEqual(len(used_gene_indices), 3)

        npt.assert_allclose(concs, [467.3333333333333, 4., 23.])
        npt.assert_allclose(used_gene_indices, [0, 1, 2])

        self.assertTrue(os.path.isfile(cache_file))
        with open(cache_file, 'r') as fd:
            content = fd.read()
            self.assertEqual(content, '1337.0,23.0,42.0\n4.0,6.0,2.0\n23.0\n')
        os.remove(cache_file)

class TestGDSFormatHandler(TestCase):
    def test_log2_ratio_format(self):
        soft = pysoft.SOFTFile('tests/data/foo.soft')
        self.assertEqual(soft.header['dataset']['dataset_value_type'], 'log2 ratio')

        gdsh = utils.GDSFormatHandler(soft)
        rows = list(gdsh.get_data())

        self.assertEqual(len(rows), 4)
        for r in rows: self.assertIsInstance(r, pysoft.parser.Row)
