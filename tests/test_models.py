from unittest import TestCase

import numpy as np
import numpy.testing as npt

import models, utils


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
        data = self.model.generate_continues_evolution(True, test_data=self.discrete_runs)
        expected = np.loadtxt('tests/data/continuous_boolean_model_run_time.txt').T

        npt.assert_allclose(data, expected)

    def test_continuous_series_over_genes(self):
        data = self.model.generate_continues_evolution(False, test_data=self.discrete_runs)
        expected = np.loadtxt('tests/data/continuous_boolean_model_run_genes.txt').T

        npt.assert_allclose(data, expected)
