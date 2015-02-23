from unittest import TestCase

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

import networkx as nx

import utils, experiment_classes


class TestGeneExpressionVariance(TestCase):
    def setUp(self):
        gdsh = utils.GDSHandler('tests/data')
        experis = gdsh.process_directory()

        self.exp = experiment_classes.GeneExpressionVariance(gdsh.common_genes, experis)

    def tearDown(self):
        pass

    def test_x(self):
        self.exp._generate_x()

        self.assertEqual(len(self.exp.x), 3)

        self.assertIn(['42', '2'], self.exp.x)
        self.assertIn(['1337', '4'], self.exp.x)
        self.assertIn(['23', '6'], self.exp.x)

        shuffled = [set(l) for l in self.exp.x_shuffled]
        self.assertIn(set(['42', '2']), shuffled)
        self.assertIn(set(['1337', '4']), shuffled)
        self.assertIn(set(['23', '6']), shuffled)

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
