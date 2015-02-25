from unittest import TestCase

import numpy as np
import numpy.testing as npt

import models, utils


class TestBaseModel(TestCase):
    def setUp(self):
        self.pure_graph = utils.GraphGenerator.get_er_graph(node_num=3, edge_prob=0.7)
        self.model = models.Model(self.pure_graph)

    def test_setup_with_aug_adja(self):
        mat = np.array([[1,0,1], [-1,0,-1], [1,0,-1]])
        model = models.Model(self.pure_graph, aug_adja=mat)

        npt.assert_allclose(mat, model.aug_adja_m)
        npt.assert_allclose(mat, self.pure_graph.aug_adja_m)

    def test_setup_without_aug_adja(self):
        self.assertIsNone(self.model.aug_adja_m)
        self.assertIsNone(self.pure_graph.aug_adja_m)

    def test_generate(self):
        with self.assertRaises(NotImplementedError):
            self.model.generate()

class TestModelMath(TestCase):
    def test_aug_adja_preservation(self):
        mat = np.array([[1,0,1], [-1,0,-1], [1,0,-1]])
        mat2 = np.array([[-1,0,1], [-1,0,1], [-1,1,1]])

        # pass to model constructor
        graph = utils.GraphGenerator.get_er_graph(node_num=3, edge_prob=0.7)
        model = models.Model(graph, aug_adja=mat)
        npt.assert_allclose(mat, model.math.get_augmented_adja_m())

        # pass to graph explicitly
        graph = utils.GraphGenerator.get_er_graph(node_num=3, edge_prob=0.7)
        graph.aug_adja_m = mat
        model = models.Model(graph)
        npt.assert_allclose(mat, model.math.get_augmented_adja_m())

        # pass to model constructor overwrites previous graph augmented adjacency matrix
        graph = utils.GraphGenerator.get_er_graph(node_num=3, edge_prob=0.7)
        graph.aug_adja_m = mat
        model = models.Model(graph, aug_adja=mat2)
        npt.assert_allclose(mat2, model.math.get_augmented_adja_m())

        # graph's augmented adjacency matrix changed later on
        graph = utils.GraphGenerator.get_er_graph(node_num=3, edge_prob=0.7)
        model = models.Model(graph)
        graph.aug_adja_m = mat
        npt.assert_allclose(mat, model.math.get_augmented_adja_m())

    def test_edge_type(self):
        graph = utils.GraphGenerator.get_er_graph(node_num=42, edge_prob=0.7)
        model = models.Model(graph)
        mat = model.math.get_augmented_adja_m(edge_type=1)
        mat[mat==1] = 0
        self.assertEqual(np.sum(mat), -len(graph))

        graph = utils.GraphGenerator.get_er_graph(node_num=42, edge_prob=0.7)
        model = models.Model(graph)
        mat = model.math.get_augmented_adja_m(edge_type=1, force_self_inhibition=False)
        mat[mat==1] = 0
        self.assertEqual(np.sum(mat), 0)

        graph = utils.GraphGenerator.get_er_graph(node_num=42, edge_prob=0.7)
        model = models.Model(graph)
        mat = model.math.get_augmented_adja_m(edge_type=-1)
        mat[mat==-1] = 0
        self.assertEqual(np.sum(mat), 0)

class TestBooleanModel(TestCase):
    def setUp(self):
        models.BooleanModel.info['cont_evo_runs'] = 300
        models.BooleanModel.info['time_window'] = 30

        graph = utils.GraphGenerator.get_er_graph(node_num=10, edge_prob=0.7)
        self.model = models.BooleanModel(graph)

        self.model.aug_adja_m = np.array([
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, -1, 0,-1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, -1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, -1]
        ])

        self.discrete_runs = np.loadtxt('tests/data/discrete_boolean_model_runs.txt')

    def test_discrete_run(self):
        x0 = [1, 1, 1, 1, 1, 0, 1, 0, 1, 0]
        data = self.model.generate_binary_time_series(initial_state=x0)

        npt.assert_allclose(data[0], [1, 1, 1, 1, 1, 0, 1, 0, 1, 0])
        npt.assert_allclose(data[1], [0, 1, 1, 0, 1, 0, 1, 0, 0, 1])
        npt.assert_allclose(data[2], [0, 0, 1, 0, 1, 0, 1, 0, 0, 1])
        for step in data[3:]: npt.assert_allclose(step, [0, 0, 1, 0, 1, 0, 1, 0, 0, 0])

    def test_continuous_series_over_time(self):
        data = self.model.generate_continuous_evolution(True, test_data=self.discrete_runs)
        expected = np.loadtxt('tests/data/continuous_boolean_model_run_time.txt').T

        npt.assert_allclose(data, expected)

    def test_continuous_series_over_genes(self):
        data = self.model.generate_continuous_evolution(False, test_data=self.discrete_runs)
        expected = np.loadtxt('tests/data/continuous_boolean_model_run_genes.txt').T

        npt.assert_allclose(data, expected)
