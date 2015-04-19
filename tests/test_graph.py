from unittest import TestCase

import os, os.path
import shutil

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

import networkx as nx

import matplotlib
matplotlib.use('agg')

import graph, models, utils


class GraphInit(TestCase):
    def setUp(self):
        self.raw_graph = nx.DiGraph()
        self.raw_graph.add_nodes_from(['1', '2', '3', '4'])
        self.raw_graph.add_edges_from([('1','2'), ('2','3'), ('3','2'), ('3','1')])
        self.graph = graph.Graph(self.raw_graph, largest=True)

    def tearDown(self):
        pass

class GraphTester(GraphInit):
    def test_initialization(self):
        self.assertEqual(len(self.graph), 3)
        self.assertEqual(list(self.graph), ['1', '2', '3'])
        npt.assert_array_equal(
            self.graph.adja_m,
            np.array([
                [0,1,0],
                [0,0,1],
                [1,1,0]
            ])
        )
        self.assertIsNone(self.graph.aug_adja_m)

        # don't just take largest subcomponent
        whole_graph = graph.Graph(self.raw_graph)

        self.assertEqual(len(whole_graph), 4)
        self.assertEqual(list(whole_graph), ['1', '2', '3', '4'])
        npt.assert_array_equal(
            whole_graph.adja_m,
            np.array([
                [0,1,0,0],
                [0,0,1,0],
                [1,1,0,0],
                [0,0,0,0]
            ])
        )
        self.assertIsNone(whole_graph.aug_adja_m)

    def test_graph_add(self):
        rg = nx.DiGraph()
        rg.add_nodes_from(['3','4','5','6'])
        rg.add_edges_from([('3','4'), ('5','4')])
        g = graph.Graph(rg)

        # test simple case
        g_sum = self.graph + g
        self.assertEqual(len(g_sum), 6)
        self.assertEqual(list(g_sum), ['1', '2', '3', '4', '5', '6'])
        self.assertEqual(set(g_sum.graph.edges()), set([('1','2'), ('2','3'), ('3','1'), ('3','2'), ('3','4'), ('5','4')]))
        npt.assert_array_equal(
            g_sum.adja_m,
            np.array([
                [0,1,0,0,0,0],
                [0,0,1,0,0,0],
                [1,1,0,1,0,0],
                [0,0,0,0,0,0],
                [0,0,0,1,0,0],
                [0,0,0,0,0,0]
            ])
        )

        # test invalid case
        with self.assertRaises(TypeError):
            foo = self.graph + 23

    def test_graph_iadd(self):
        rg = nx.DiGraph()
        rg.add_nodes_from(['42'])
        g = graph.Graph(rg)

        g += self.graph

        self.assertEqual(len(g), 4)
        self.assertEqual(list(g), ['1', '2', '3', '42'])
        self.assertEqual(set(g.graph.edges()), set([('1','2'), ('2','3'), ('3','1'), ('3','2')]))
        npt.assert_array_equal(
            g.adja_m,
            np.array([
                [0,1,0,0],
                [0,0,1,0],
                [1,1,0,0],
                [0,0,0,0],
            ])
        )

        with self.assertRaises(TypeError):
            g += 42

    def test_graph_conversion_on_composition(self):
        drg = nx.DiGraph()
        drg.add_nodes_from([3,4,5,6])
        drg.add_edges_from([(3,4), (5,4)])
        dg = graph.Graph(drg)
        self.assertIsInstance(dg.graph, nx.DiGraph)

        mrg = nx.MultiDiGraph()
        mrg.add_nodes_from([1,2])
        mrg.add_edges_from([(1,2), (2,1)])
        mg = graph.Graph(mrg)
        self.assertIsInstance(mg.graph, nx.MultiDiGraph)

        sg = dg + mg
        self.assertIsInstance(sg.graph, nx.MultiDiGraph)

    def test_graph_reduction(self):
        rg = nx.DiGraph()
        rg.add_nodes_from(['1', '2'])
        g = graph.Graph(rg)

        self.graph.reduce_to(g)

        self.assertEqual(len(self.graph), 2)
        self.assertEqual(list(self.graph), ['1', '2'])
        self.assertEqual(self.graph.graph.edges(), [('1','2')])

class TestGraphIO(GraphInit):
    def test_adja_dumping(self):
        adja_file = 'adja_m_testing.txt'
        self.graph.io.dump_adjacency_matrix(adja_file)

        with open(adja_file, 'r') as fd:
            mat = fd.read()
            self.assertEqual(mat, '0 1 0\n0 0 1\n1 1 0\n')

        os.remove(adja_file)

    def test_node_dumping(self):
        node_file = 'node_testing.txt'
        self.graph.io.dump_node_names(node_file)

        with open(node_file, 'r') as fd:
            nodes = fd.read()
            self.assertEqual(nodes, '1\n2\n3\n')

        os.remove(node_file)

    def test_matplotlib_visualization(self):
        image_file = 'graph_testing.png'

        self.graph.io.visualize(image_file, use_R=False)
        self.assertTrue(os.path.isfile(image_file))

        os.remove(image_file)

    def test_delegation(self):
        utils.DataHandler.backup_dir = 'conc_bak_test'
        conc_file = 'tests/data/foo.soft'
        conc_dir = 'tests/data/'

        raw_graph = nx.DiGraph()
        raw_graph.add_nodes_from(['aaea', 'aaeb', 'haba', 'zuzu'])
        trn = graph.Graph(raw_graph)

        res1 = utils.DataHandler.load_concentrations(trn, conc_file, conc_range=[0])
        res2 = trn.io.load_concentrations(conc_file)
        self.assertEqual(res1, res2)

        res1 = utils.DataHandler.load_averaged_concentrations(trn, conc_dir, conc_range=[0])
        res2 = trn.io.load_averaged_concentrations(conc_dir, conc_range=[0])
        self.assertEqual(res1, res2)

        shutil.rmtree(utils.DataHandler.backup_dir)

    def test_method_selection(self):
        microarray_file = 'tests/data/foo.soft'
        rnaseq_file = 'tests/data/rnaseq.count'

        raw_graph = nx.DiGraph()
        raw_graph.add_nodes_from(['aaea', 'aaeb', 'haba', 'zuzu'])
        trn = graph.Graph(raw_graph)

        ma_exp = trn.io.load_concentrations(microarray_file)
        rs_exp = trn.io.load_concentrations(rnaseq_file)

        self.assertEqual(list(ma_exp.get_columns()), ['GSM37063', 'GSM37064'])
        self.assertEqual(list(rs_exp.get_columns()), ['RNAseq'])

class TestGraphSystem(GraphInit):
    def test_unassigned_model(self):
        with self.assertRaises(RuntimeError):
            self.graph.system.simulate()

class TestGraphMath(GraphInit):
    def test_perron_frobenius(self):
        mat = np.array([[3,0], [8,-1]])
        pf = np.array([1,2])

        npt.assert_allclose(self.graph.math.get_perron_frobenius(mat=mat), pf/npl.norm(pf))

    def test_significance_test(self):
        mat = np.array([[3,0], [8,-1]])
        pf = np.array([1,2])

        self.assertTrue(self.graph.math.test_significance(3, pf, mat=mat))
        self.assertFalse(self.graph.math.test_significance(4, pf, mat=mat))
        self.assertFalse(self.graph.math.test_significance(3, np.array([4,-1]), mat=mat))

    def test_power_iteration(self):
        mat = np.array([[1,-0.5], [0,4]])
        eival = 4
        pf = np.array([-0.164399, 0.986394])

        npt.assert_allclose(pf, self.graph.math.apply_power_iteration(eival, mat=mat))

    def test_pagerank(self):
        res = self.graph.math.get_pagerank()

        self.assertEqual(len(self.graph), len(res))

    def test_degree_distribution(self):
        npt.assert_allclose(self.graph.math.get_degree_distribution(), np.array([0.42640143, 0.63960215, 0.63960215]))

    def test_aug_adja_m_generation(self):
        rg = nx.DiGraph()
        rg.add_nodes_from([1, 2])
        rg.add_edges_from([(1,2)])


        uam = np.array([[-1,-1], [-1,-1]])
        g = graph.Graph(rg)
        g.aug_adja_m = uam.copy()
        g.system.simulate(models.BooleanModel, runs=4)
        npt.assert_allclose(g.aug_adja_m, uam)
        npt.assert_allclose(g.aug_adja_m, g.system.used_model.aug_adja_m)

        g = graph.Graph(rg)
        g.system.simulate(models.BooleanModel, runs=4, aug_adja=uam.copy())
        npt.assert_allclose(g.system.used_model.aug_adja_m, uam)
        npt.assert_allclose(g.aug_adja_m, g.system.used_model.aug_adja_m)


        g = graph.Graph(rg)
        g.system.simulate(models.BooleanModel, runs=4, edge_type=-1, force_self_inhibition=False)
        npt.assert_allclose(g.system.used_model.aug_adja_m, np.array([[0,-1], [0,0]], dtype=np.int8))
        npt.assert_allclose(g.aug_adja_m, g.system.used_model.aug_adja_m)

        # reuse previous aa matrix
        g.system.simulate(runs=4)
        npt.assert_allclose(g.system.used_model.aug_adja_m, np.array([[0,-1], [0,0]], dtype=np.int8))
        npt.assert_allclose(g.aug_adja_m, g.system.used_model.aug_adja_m)
