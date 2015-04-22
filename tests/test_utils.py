from unittest import TestCase

import os.path, shutil

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

import scipy.stats as scits
import networkx as nx

import pysoft

import utils, errors, models, graph


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

    def test_interquartile_variance(self):
        data = [102, 104, 105, 107, 108, 109, 110, 112, 115, 116, 118]

        res = utils.get_interquartile_variance(data)
        npt.assert_approx_equal(res, 9.3877551)

        res = utils.get_interquartile_variance(data, pop_range=[0, 100])
        npt.assert_approx_equal(res, 24.231404)

    def test_max_real_entry_detection(self):
        vec = [4-5j, 1, 5, 2, 6, 3, 7+1j, 6+10j]

        self.assertEqual(utils.get_max_entry_index(vec), 6)
        self.assertEqual(utils.get_max_entry_index(vec, real_entries_only=True), 4)

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

    def test_random_graph_er_network_preservation(self):
        graph = utils.GraphGenerator.get_random_graph(42, 30, 50)
        new_graph = utils.GraphGenerator.get_random_graph(graph, 60, 13)

        self.assertEqual(len(new_graph), 42)
        self.assertIsInstance(new_graph.graph, nx.DiGraph)

        self.assertEqual(graph.graph.nodes(), new_graph.graph.nodes())
        self.assertNotEqual(
            graph.aug_adja_m.tolist(),
            new_graph.aug_adja_m.tolist()
        )

        pos_m = new_graph.aug_adja_m.copy()
        pos_m[pos_m != 1] = 0
        self.assertEqual(np.sum(pos_m), 60)

        neg_m = new_graph.aug_adja_m.copy()
        neg_m[neg_m != -1] = 0
        self.assertEqual(np.sum(neg_m), -13)

    def test_random_graph_scalefree_network_preservation(self):
        graph = utils.GraphGenerator.get_scalefree_graph(100)
        edge_num = len(graph.graph.edges())

        new_graph = utils.GraphGenerator.get_random_graph(
            42,
            round(1/2 * edge_num), edge_num - round(1/2 * edge_num)
        )
        new_graph2 = utils.GraphGenerator.get_random_graph(
            new_graph,
            round(1/3 * edge_num), edge_num - round(1/3 * edge_num)
        )

        self.assertEqual(
            set(new_graph.graph.edges()),
            set(new_graph2.graph.edges())
        )

    def test_random_graph_network_preservation_with_model(self):
        models.BooleanModel.info['cont_evo_runs'] = 5


        graph = utils.GraphGenerator.get_random_graph(42, 30, 50)
        graph.system.simulate(models.BooleanModel)
        amat = graph.system.used_model.math.get_augmented_adja_m()

        self.assertEqual(
            amat.tolist(),
            graph.aug_adja_m.tolist()
        )


        new_graph = utils.GraphGenerator.get_random_graph(graph, 60, 13)
        self.assertIsNone(new_graph.system.used_model)

        new_graph.system.simulate(models.BooleanModel)
        new_amat = new_graph.system.used_model.math.get_augmented_adja_m()

        self.assertEqual(
            new_amat.tolist(),
            new_graph.aug_adja_m.tolist()
        )

        self.assertNotEqual(
            amat.tolist(),
            new_amat.tolist()
        )

    def test_random_graph_invalid_preservation(self):
        with self.assertRaises(RuntimeError):
            new_graph = utils.GraphGenerator.get_random_graph('foo', 60, 13)

    def test_random_graph_only_activating(self):
        graph = utils.GraphGenerator.get_random_graph(5, 10)

        self.assertEqual(len(graph), 5)
        self.assertIsInstance(graph.graph, nx.DiGraph)

        pos_m = graph.aug_adja_m.copy()
        pos_m[pos_m != 1] = 0
        self.assertEqual(np.sum(pos_m), 10)

        neg_m = graph.aug_adja_m.copy()
        neg_m[neg_m != -1] = 0
        self.assertEqual(np.sum(neg_m), 0)

    def test_random_graph_only_inhibiting(self):
        graph = utils.GraphGenerator.get_random_graph(5, 0, 10)

        self.assertEqual(len(graph), 5)
        self.assertIsInstance(graph.graph, nx.DiGraph)

        pos_m = graph.aug_adja_m.copy()
        pos_m[pos_m != 1] = 0
        self.assertEqual(np.sum(pos_m), 0)

        neg_m = graph.aug_adja_m.copy()
        neg_m[neg_m != -1] = 0
        self.assertEqual(np.sum(neg_m), -10)

    def test_er_graph(self):
        graph = utils.GraphGenerator.get_er_graph(100, 0.66)

        self.assertEqual(len(graph), 100)
        self.assertIsInstance(graph.graph, nx.DiGraph)

    def test_scalefree_graph(self):
        graph = utils.GraphGenerator.get_scalefree_graph(100)

        self.assertEqual(len(graph), 100)
        self.assertIsInstance(graph.graph, nx.DiGraph)

    def test_trn_graph(self):
        trn_file = 'tests/data/trn_network.txt'
        gpn_file = 'tests/data/gene_proximity_network.txt'

        # without gpn is tested for file_parser, test TRN/GPN concatenation
        graph = utils.GraphGenerator.get_regulatory_graph(trn_file, gpn_file, base_window=10, reduce_gpn=False)

        self.assertIsInstance(graph.graph, nx.MultiDiGraph)
        self.assertEqual(len(graph), 8)
        self.assertEqual(list(graph), ['aaea', 'aaeb', 'arga', 'cydc', 'fnge', 'mepm', 'yaaa', 'zuzu'])

        self.assertEqual(set(graph.graph.edges()), set([
            ('yaaa', 'cydc'), ('cydc', 'yaaa'),
            ('fnge', 'mepm'), ('mepm', 'fnge'),
            ('fnge', 'arga'), ('arga', 'fnge'),
            ('arga', 'yaaa'), ('yaaa', 'arga'),
            ('aaea', 'aaeb'), ('aaea', 'zuzu'),
            ('aaeb', 'aaea'), ('zuzu', 'zuzu'),
            ('zuzu', 'aaeb')
        ]))

    def test_gpn_graph(self):
        gpn_file = 'tests/data/gene_proximity_network.txt'

        # circular genome
        graph = utils.GraphGenerator.get_gene_proximity_network(gpn_file, base_window=10)
        self.assertEqual(set(graph.graph.edges()), set([
            ('yaaa', 'cydc'), ('cydc', 'yaaa'),
            ('fnge', 'mepm'), ('mepm', 'fnge'),
            ('fnge', 'arga'), ('arga', 'fnge'),
            ('arga', 'yaaa'), ('yaaa', 'arga'),
        ]))

        # two stranded genome
        graph = utils.GraphGenerator.get_gene_proximity_network(gpn_file, base_window=10, origin=72)
        self.assertEqual(set(graph.graph.edges()), set([
            ('yaaa', 'arga'), ('arga', 'yaaa'),
            ('fnge', 'mepm'), ('mepm', 'fnge')
        ]))

class TestStatsHandler(TestCase):
    def setUp(self):
        self.v = list(range(0, 100))
        self.vex = np.exp(self.v)

    def test_pearson_correlation(self):
        utils.StatsHandler.FUNC = scits.pearsonr

        c, p = utils.StatsHandler().correlate(self.v, self.vex)
        npt.assert_almost_equal(c, 0.25, decimal=2)
        npt.assert_almost_equal(p, 0.01, decimal=2)

        c, p = utils.StatsHandler().correlate(self.v, list(reversed(self.vex)))
        npt.assert_almost_equal(c, -0.25, decimal=2)
        npt.assert_almost_equal(p, 0.01, decimal=2)

        c, p, mi, ma = utils.StatsHandler().correlate(self.v, self.vex, compute_bands=True)
        npt.assert_almost_equal(c, 0.25, decimal=2)
        npt.assert_almost_equal(p, 0.01, decimal=2)
        self.assertTrue(mi > -1)
        self.assertTrue(ma < 1)

    def test_spearman_correlation(self):
        utils.StatsHandler.FUNC = scits.spearmanr

        c, p = utils.StatsHandler().correlate(self.v, self.vex)
        npt.assert_almost_equal(c, 1., decimal=2)
        npt.assert_almost_equal(p, 0., decimal=2)

        c, p = utils.StatsHandler().correlate(self.v, list(reversed(self.vex)))
        npt.assert_almost_equal(c, -1., decimal=2)
        npt.assert_almost_equal(p, 0., decimal=2)

        c, p, mi, ma = utils.StatsHandler().correlate(self.v, self.vex, compute_bands=True)
        npt.assert_almost_equal(c, 1., decimal=2)
        npt.assert_almost_equal(p, 0., decimal=2)
        self.assertTrue(mi > -1)
        self.assertTrue(ma < 1)

class TestCacheHandler(TestCase):
    def setUp(self):
        utils.CacheHandler.cache_directory = 'plot_data_testing'

        self.func = lambda x: x

    def tearDown(self):
        try:
            shutil.rmtree(utils.CacheHandler.cache_directory)
        except FileNotFoundError:
            pass

    def test_data_storage_2_args(self):
        res = utils.CacheHandler.store_plot_data('The Neverending Story', self.func, 'x axis stuff', 'important data', foo='bar')

        self.assertEqual(len(res), 5)
        self.assertEqual(len(res['info']), 1)

        self.assertEqual(res['args'], {'foo': 'bar'})

        self.assertEqual(res['title'], 'The Neverending Story')
        self.assertEqual(res['info']['function'], '<lambda>')

        self.assertEqual(res['x_label'], 'x axis stuff')
        self.assertEqual(res['data'], 'important data')

        self.assertTrue(os.path.isfile(os.path.join(utils.CacheHandler.cache_directory, 'The_Neverending_Story__raw.dat')))

    def test_data_storage_3_args(self):
        res = utils.CacheHandler.store_plot_data('The Neverending Story', self.func, 'x axis stuff', 'y axis stuff', 'important data')

        self.assertEqual(len(res), 6)
        self.assertEqual(len(res['info']), 1)
        self.assertEqual(len(res['args']), 0)

        self.assertEqual(res['title'], 'The Neverending Story')
        self.assertEqual(res['info']['function'], '<lambda>')

        self.assertEqual(res['x_label'], 'x axis stuff')
        self.assertEqual(res['y_label'], 'y axis stuff')
        self.assertEqual(res['data'], 'important data')

        self.assertTrue(os.path.isfile(os.path.join(utils.CacheHandler.cache_directory, 'The_Neverending_Story__raw.dat')))

    def test_data_storage_4_args(self):
        res = utils.CacheHandler.store_plot_data('The Neverending Story', self.func, 'x axis stuff', 'x data', 'y axis stuff', 'y data')

        self.assertEqual(len(res), 7)
        self.assertEqual(len(res['info']), 1)
        self.assertEqual(len(res['args']), 0)

        self.assertEqual(res['title'], 'The Neverending Story')
        self.assertEqual(res['info']['function'], '<lambda>')

        self.assertEqual(res['x_label'], 'x axis stuff')
        self.assertEqual(res['y_label'], 'y axis stuff')
        self.assertEqual(res['x_data'], 'x data')
        self.assertEqual(res['y_data'], 'y data')
        self.assertTrue(os.path.isfile(os.path.join(utils.CacheHandler.cache_directory, 'The_Neverending_Story__raw.dat')))

    def test_data_storage_multiplicator_model(self):
        res = utils.CacheHandler.store_plot_data('The Neverending Story', self.func, 'x axis stuff', 'important data', model=models.MultiplicatorModel)

        self.assertEqual(len(res), 5)
        self.assertEqual(len(res['info']), 2)
        self.assertEqual(len(res['args']), 0)

        self.assertEqual(res['title'], 'The Neverending Story')
        self.assertEqual(res['info']['function'], '<lambda>')
        self.assertEqual(res['info']['name'], 'Multiplicator Model')

        self.assertEqual(res['x_label'], 'x axis stuff')
        self.assertEqual(res['data'], 'important data')

        self.assertTrue(os.path.isfile(os.path.join(utils.CacheHandler.cache_directory, 'The_Neverending_Story__%s.dat' % models.MultiplicatorModel.hash())))

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
        res = self.gdsh.parse_file(gds_file, conc_range=[0])

        self.assertEqual(
            res.data, {
                'GSM37063': {
                    'aaea': 42.,
                    'aaeb': 2.,
                    'zuzu': 23.
                }
            }
        )
        self.assertEqual(res.get_genes(), ['aaea', 'aaeb', 'zuzu'])
        self.assertEqual(res.filename, 'foo.soft')

    def testInvalidFile(self):
        gds_file = 'qux.soft'

        with self.assertRaises(errors.InvalidGDSFormatError):
            res = self.gdsh.parse_file(gds_file)

    def test_all_genes(self):
        res = self.gdsh.process_directory(conc_range=[0, 'GSM37064'])

        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].data['GSM37063'], {'aaea': 1337., 'aaeb': 4.})
        self.assertEqual(res[0].data['GSM37064'], {})
        self.assertEqual(res[1].data['GSM37063'], {'aaea': 23., 'aaeb': 6.})
        self.assertEqual(res[1].data['GSM37064'], {})
        self.assertEqual(res[2].data['GSM37063'], {'aaea': 42., 'aaeb': 2., 'zuzu': 23.})
        self.assertEqual(res[2].data['GSM37064'], {'aaea': 43., 'aaeb': 3., 'haba': 1, 'zuzu': 24.})

        self.assertEqual(len(self.gdsh.all_genes), 4)
        self.assertEqual(len(self.gdsh.quasi_genes), 3)
        self.assertEqual(len(self.gdsh.common_genes), 2)

    def test_common_genes(self):
        res = self.gdsh.process_directory(only_common_genes=True, conc_range=['GSM37063', 1])

        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].data['GSM37063'], {'aaea': 1337., 'aaeb': 4.})
        self.assertEqual(res[1].data['GSM37063'], {'aaea': 23., 'aaeb': 6.})
        self.assertEqual(res[2].data['GSM37063'], {'aaea': 42., 'aaeb': 2.})
        self.assertEqual(res[2].data['GSM37064'], {'aaea': 43., 'aaeb': 3.})

        self.assertEqual(len(self.gdsh.all_genes), 4)
        self.assertEqual(len(self.gdsh.quasi_genes), 3)
        self.assertEqual(len(self.gdsh.common_genes), 2)

class TestDataHandler(TestCase):
    def setUp(self):
        utils.DataHandler.backup_dir = 'conc_bak_test'

        network_file = 'tests/data/trn_network.txt'
        self.graph = utils.GraphGenerator.get_regulatory_graph(network_file)

        self.average_cache_file = 'RL_av_data_testing.csv'

    def tearDown(self):
        try:
            shutil.rmtree(utils.DataHandler.backup_dir)
        except FileNotFoundError:
            pass

        try:
            os.remove(self.average_cache_file)
        except FileNotFoundError:
            pass

    def test_empty_file(self):
        conc_file = 'tests/data/empty.soft.spec'
        exp = utils.DataHandler.load_concentrations(self.graph, conc_file)

        self.assertEqual(len(exp.data), 0)
        self.assertEqual(len(exp.get_genes()), 0)

    def test_single_file(self):
        conc_file = 'tests/data/foo.soft'
        exp = utils.DataHandler.load_concentrations(self.graph, conc_file, conc_range=[1])

        self.assertTrue(os.path.isdir(utils.DataHandler.backup_dir))
        self.assertTrue(os.path.isfile(os.path.join(utils.DataHandler.backup_dir, 'conc_foo.soft.bak.npy')))

        self.assertEqual(exp.data['GSM37064'], {'aaea': 43., 'aaeb': 3., 'zuzu': 24.})
        self.assertEqual(exp.get_genes(), ['aaea', 'aaeb', 'zuzu'])

    def test_file_averaging(self):
        conc_dir = 'tests/data/'
        exp = utils.DataHandler.load_averaged_concentrations(self.graph, conc_dir, cache_file=self.average_cache_file, conc_range=[0, 1])

        self.assertEqual(len(exp.data['average']), 3)
        self.assertEqual(len(exp.get_genes()), 3)

        self.assertEqual(exp.data['average'], {'aaea': 361.25, 'aaeb': 3.75, 'zuzu': 23.5})
        self.assertEqual(exp.get_genes(), ['aaea', 'aaeb', 'zuzu'])

        expected_content = 'aaea,1337.0,23.0,42.0,43.0\naaeb,4.0,6.0,2.0,3.0\nzuzu,23.0,24.0\n'
        self.assertTrue(os.path.isfile(self.average_cache_file))
        with open(self.average_cache_file, 'r') as fd:
            content = fd.read()
            self.assertEqual(content, expected_content)

        # check if multiple runs don't screw up data file
        exp = utils.DataHandler.load_averaged_concentrations(self.graph, conc_dir, cache_file=self.average_cache_file, conc_range=[0, 1])
        self.assertTrue(os.path.isfile(self.average_cache_file))
        with open(self.average_cache_file, 'r') as fd:
            content = fd.read()
            self.assertEqual(content, expected_content)

    def test_partial_file_averaging(self):
        conc_dir = 'tests/data/'
        partial_network_file = 'tests/data/partial_trn_network.txt'

        partial_graph = utils.GraphGenerator.get_regulatory_graph(partial_network_file)
        exp = utils.DataHandler.load_averaged_concentrations(partial_graph, conc_dir, conc_range=[0])

        self.assertEqual(len(exp.data['average']), 2)
        self.assertEqual(len(exp.get_genes()), 2)

        self.assertEqual(exp.data['average'], {'aaea': 467.3333333333333, 'aaeb': 4.})
        self.assertEqual(exp.get_genes(), ['aaea', 'aaeb'])

    def test_rnaseq(self):
        count_file = 'tests/data/rnaseq.count'
        exp = utils.DataHandler.load_rnaseq_data(self.graph, count_file)

        self.assertEqual(exp.data['RNAseq'], {'aaea': 13, 'zuzu': 42})
        self.assertEqual(exp.get_genes(), ['aaea', 'zuzu'])

class TestGDSFormatHandler(TestCase):
    def test_log2_ratio_format(self):
        soft = pysoft.SOFTFile('tests/data/foo.soft')
        self.assertEqual(soft.header['dataset']['dataset_value_type'], 'log2 ratio')

        gdsh = utils.GDSFormatHandler(soft)
        rows = list(gdsh.get_data())

        self.assertEqual(len(rows), 4)
        for r in rows: self.assertIsInstance(r, pysoft.parser.Row)

    def test_unknown_format(self):
        soft = pysoft.SOFTFile('tests/data/qux.soft')
        self.assertEqual(soft.header['dataset']['dataset_value_type'], 'unsupported format')

        # raise error
        gdsh = utils.GDSFormatHandler(soft)
        with self.assertRaises(errors.InvalidGDSFormatError):
            rows = list(gdsh.get_data())

        # deal with it
        gdsh = utils.GDSFormatHandler(soft, throw_on_unknown_format=False)
        rows = list(gdsh.get_data())

        self.assertEqual(len(rows), 1)
        for r in rows: self.assertIsInstance(r, pysoft.parser.Row)

    def test_row_formatter(self):
        soft = pysoft.SOFTFile('tests/data/foo.soft')
        gdsh = utils.GDSFormatHandler(soft)

        row = soft.data[0]
        self.assertEqual(list(row), ['bar', 'aaea', 42, 43])

        res = gdsh.transform_row(row, np.log)
        self.assertEqual(list(res), ['bar', 'aaea', np.log(42), np.log(43)])

        res = gdsh.transform_row(row, lambda x: x+10)
        self.assertEqual(list(res), ['bar', 'aaea', 52, 53])

    def test_column_selector(self):
        soft = pysoft.SOFTFile('tests/data/col_case.soft.fubar')
        gdsh = utils.GDSFormatHandler(soft)

        cols = gdsh.get_useful_columns()

        self.assertEqual(len(cols), 8)

        self.assertTrue('GSM37063' in cols)
        self.assertTrue('GSM37064' in cols)
        self.assertTrue('GSM37065' in cols)
        self.assertTrue('GSM37066' in cols)
        self.assertTrue('GSM37067' in cols)
        self.assertTrue('GSM37068' in cols)
        self.assertTrue('GSM37069' in cols)
        # GSM37070 is mutant
        self.assertTrue('GSM37071' in cols)

class TestGDSParseResult(TestCase):
    def test_initialization(self):
        res = utils.GDSParseResult(['foo', 'bar', 'baz', 'qux'])
        self.assertEqual(sorted(res.data.keys()), ['bar', 'baz', 'foo', 'qux'])
        self.assertEqual(list(res.get_columns()), ['bar', 'baz', 'foo', 'qux'])

    def test_gene_getter(self):
        res = utils.GDSParseResult()
        res.data = {'foo': {'a': 1, 'b': 2}}

        self.assertEqual(list(res.get_genes_in_col('foo')), ['a', 'b'])

    def test_gene_addition(self):
        res = utils.GDSParseResult()

        res.add_gene('z')
        res.add_gene('a')
        res.add_gene('b')
        self.assertEqual(res.get_genes(), ['a', 'b', 'z'])

        res.add_genes(['d', 'c'])
        self.assertEqual(res.get_genes(), ['a', 'b', 'c', 'd', 'z'])

        res.clear_genes()
        res.add_genes(['d', 'c'])
        self.assertEqual(res.get_genes(), ['c', 'd'])

    def test_equality_check(self):
        res1 = utils.GDSParseResult(['foo'])
        res2 = utils.GDSParseResult(['foo'])
        self.assertEqual(res1, res2)

        res2.add_filename('fubar.soft')
        self.assertNotEqual(res1, res2)

    def test_filename_setter(self):
        res = utils.GDSParseResult()

        res.add_filename('foo')
        self.assertEqual(res.filename, 'foo')

        res.add_filename('bar')
        self.assertEqual(res.filename, 'foo,bar')

        res.add_filename('baz')
        self.assertEqual(res.filename, 'foo,bar,baz')

    def test_data_getter(self):
        res = utils.GDSParseResult()
        res.data = {'foo': {'a': 1, 'b': 2}, 'bar': {'a': 10, 'b': 20}}

        gener = list(res.get_data())
        self.assertEqual(len(gener), 2)
        self.assertEqual(gener[0], ('bar', [10, 20]))
        self.assertEqual(gener[1], ('foo', [1, 2]))

    def test_input_trimming(self):
        raw_graph = nx.DiGraph()
        raw_graph.add_nodes_from(['a', 'b', 'c', 'd'])
        g = graph.Graph(raw_graph)

        res = utils.GDSParseResult()
        res.data = {'foo': {'b': 2, 'd': 4}}

        inp = [4, 3, 2, 1]
        trimmed = res.trim_input(inp, g, 'foo')

        self.assertEqual(trimmed, [3, 1])

    def test_common_gene_extraction(self):
        res = utils.GDSParseResult()
        res.data = {'foo': {'a': 1, 'b': 2}, 'bar': {'b': 10, 'c': 20}}

        cgenes = res.get_common_genes()
        self.assertEqual(len(cgenes), 1)
        self.assertEqual(cgenes, ['b'])
