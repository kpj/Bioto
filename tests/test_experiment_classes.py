from unittest import TestCase

import numpy as np
import numpy.random as npr
import numpy.testing as npt

import networkx as nx

import matplotlib
matplotlib.use('Agg')

import utils, experiment_classes


class TestBaseExperiment(TestCase):
    def setUp(self):
        self.exp = experiment_classes.Experiment()

    def test_conduct(self):
        with self.assertRaises(NotImplementedError):
            self.exp.conduct()

class TestGeneExpressionVariance(TestCase):
    def setUp(self):
        gdsh = utils.GDSHandler('tests/data')
        experis = gdsh.process_directory(only_common_genes=True)

        self.exp = experiment_classes.GeneExpressionVariance(gdsh.common_genes, experis)

    def tearDown(self):
        pass

    def test_x(self):
        self.exp._generate_x()

        self.assertEqual(len(self.exp.x), 3)

        self.assertIn([42., 2.], self.exp.x)
        self.assertIn([1337., 4.], self.exp.x)
        self.assertIn([23., 6.], self.exp.x)

        shuffled = [set(l) for l in self.exp.x_shuffled]
        self.assertIn(set([42., 2.]), shuffled)
        self.assertIn(set([1337., 4.]), shuffled)
        self.assertIn(set([23., 6.]), shuffled)

    def test_x_shuffled_experiments(self):
        npr.seed(42)
        self.exp._generate_x(shuffle_experiment_order=True)

        self.assertEqual(len(self.exp.x), 3)

        self.assertEqual(self.exp.x[0], [23., 6.])
        self.assertEqual(self.exp.x[1], [1337., 4.])
        self.assertEqual(self.exp.x[2], [42., 2.])

        shuffled = [set(l) for l in self.exp.x_shuffled]
        self.assertEqual(len(shuffled), 3)
        self.assertEqual(shuffled[0], set([23., 6.]))
        self.assertEqual(shuffled[1], set([1337., 4.]))
        self.assertEqual(shuffled[2], set([42., 2.]))


        self.exp.x = []
        self.exp.x_shuffled = []


        npr.seed(1337)
        self.exp._generate_x(shuffle_experiment_order=True)

        self.assertEqual(len(self.exp.x), 3)

        self.assertEqual(self.exp.x[0], [42., 2.])
        self.assertEqual(self.exp.x[1], [23., 6.])
        self.assertEqual(self.exp.x[2], [1337., 4.])

        shuffled = [set(l) for l in self.exp.x_shuffled]
        self.assertEqual(len(shuffled), 3)
        self.assertEqual(shuffled[0], set([42., 2.]))
        self.assertEqual(shuffled[1], set([23., 6.]))
        self.assertEqual(shuffled[2], set([1337., 4.]))

    def test_y(self):
        self.exp._generate_x()
        self.exp._generate_y()

        self.assertEqual(len(self.exp.y), 2)

        npt.assert_allclose(self.exp.y[0], [680, 5])
        npt.assert_allclose(self.exp.y[1], [467.333333, 4])

    def test_variances(self):
        self.exp._generate_x()
        self.exp._generate_y()
        self.exp._compute_variances()

        self.assertEqual(len(self.exp.variances), 2)

        npt.assert_allclose(self.exp.variances, [113906.25, 53669.444367222219])

    def test_variances_shuffled_experiments(self):
        self.exp.conduct(shuffle_experiment_order=True)

        npt.assert_approx_equal(self.exp.variances[-1], 53669.444367222219)

    def test_conduct(self):
        self.exp._generate_x()
        self.exp._generate_y()
        self.exp._compute_variances()
        res1 = self.exp.variances.copy()

        # reset experiment
        self.exp.x = []
        self.exp.x_shuffled = []
        self.exp.y = []
        self.exp.y_shuffled = []
        self.exp.variances = []
        self.exp.variances_shuffled = []

        self.exp.conduct()
        res2 = self.exp.variances.copy()

        npt.assert_allclose(res1, res2)
