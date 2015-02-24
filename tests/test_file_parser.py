from unittest import TestCase

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

import networkx as nx

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
