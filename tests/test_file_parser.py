from unittest import TestCase

import numpy as np
import numpy.testing as npt

import networkx as nx

import file_parser


class TestParser(TestCase):
    def test_concentration_parser_single(self):
        gds_file = 'tests/data/foo.soft'

        res = file_parser.parse_concentration(gds_file, conc_range=[0])
        self.assertEqual(len(res.data['GSM37063']), 3)
        self.assertEqual(res.data['GSM37063']['aaea'], 42.)
        self.assertEqual(res.data['GSM37063']['aaeb'], 2.)
        self.assertEqual(res.data['GSM37063']['zuzu'], 23.)
        self.assertEqual(res.get_genes(), ['aaea', 'aaeb', 'zuzu'])
        self.assertEqual(res.filename, 'foo.soft')

        res = file_parser.parse_concentration(gds_file, conc_range=[1])
        self.assertEqual(len(res.data['GSM37064']), 4)
        self.assertEqual(res.data['GSM37064']['aaea'], 43.)
        self.assertEqual(res.data['GSM37064']['aaeb'], 3.)
        self.assertEqual(res.data['GSM37064']['haba'], 1.)
        self.assertEqual(res.data['GSM37064']['zuzu'], 24.)
        self.assertEqual(res.get_genes(), ['aaea', 'aaeb', 'haba', 'zuzu'])
        self.assertEqual(res.filename, 'foo.soft')

    def test_concentration_parser_column_selection(self):
        gds_file = 'tests/data/col_case.soft.fubar'
        res = file_parser.parse_concentration(gds_file)

        self.assertEqual(len(res.data), 8)

        self.assertEqual(res.data['GSM37063']['aaea'], 1.)
        self.assertEqual(res.data['GSM37064']['aaea'], 2.)
        self.assertEqual(res.data['GSM37065']['aaea'], 3.)
        self.assertEqual(res.data['GSM37066']['aaea'], 4.)
        self.assertEqual(res.data['GSM37067']['aaea'], 5.)
        self.assertEqual(res.data['GSM37068']['aaea'], 6.)
        self.assertEqual(res.data['GSM37069']['aaea'], 7.)
        # GSM37070 is mutant
        self.assertEqual(res.data['GSM37071']['aaea'], 9.)

        self.assertEqual(res.get_genes(), ['aaea'])
        self.assertEqual(res.filename, 'col_case.soft.fubar')

    def test_concentration_parser_multiple(self):
        gds_file = 'tests/data/foo.soft'
        res = file_parser.parse_concentration(gds_file, conc_range=[0, 1])

        self.assertEqual(len(res.data['GSM37063']), 3)
        self.assertEqual(res.data['GSM37063']['aaea'], 42.)
        self.assertEqual(res.data['GSM37063']['aaeb'], 2.)
        self.assertEqual(res.data['GSM37063']['zuzu'], 23.)

        self.assertEqual(len(res.data['GSM37064']), 4)
        self.assertEqual(res.data['GSM37064']['aaea'], 43.)
        self.assertEqual(res.data['GSM37064']['aaeb'], 3.)
        self.assertEqual(res.data['GSM37064']['haba'], 1.)
        self.assertEqual(res.data['GSM37064']['zuzu'], 24.)

        self.assertEqual(res.get_genes(), ['aaea', 'aaeb', 'haba', 'zuzu'])

    def test_conc_range(self):
        gds_file = 'tests/data/foo.soft'

        data1 = file_parser.parse_concentration(gds_file, conc_range=[0])
        data2 = file_parser.parse_concentration(gds_file, conc_range=['GSM37063'])

        self.assertEqual(data1, data2)

    def test_regulation_parser(self):
        network_file = 'tests/data/trn_network.txt'
        data = file_parser.parse_regulation_file(network_file)

        self.assertEqual(len(data), 3)
        self.assertEqual(data['aaea'], [('aaeb', '+'), ('zuzu', '+')])
        self.assertEqual(data['aaeb'], [('aaea', '-')])
        self.assertEqual(data['zuzu'], [('zuzu', '-'), ('aaeb', '+-')])

    def test_trn_generation(self):
        network_file = 'tests/data/trn_network.txt'
        graph = file_parser.generate_tf_gene_regulation(network_file)

        self.assertIsInstance(graph, nx.DiGraph)

        self.assertEqual(len(graph), 3)
        self.assertIn('aaea', graph)
        self.assertIn('aaeb', graph)
        self.assertIn('zuzu', graph)

        self.assertEqual(len(graph.edges()), 5)
        self.assertIn(('aaea', 'aaeb'), graph.edges())
        self.assertIn(('aaea', 'zuzu'), graph.edges())
        self.assertIn(('aaeb', 'aaea'), graph.edges())
        self.assertIn(('zuzu', 'zuzu'), graph.edges())
        self.assertIn(('zuzu', 'aaeb'), graph.edges())

    def test_gene_proximity_parser(self):
        gpng = file_parser.GPNGenerator('tests/data/gene_proximity_network.txt')

        self.assertEqual(gpng.max_right, 100)
        self.assertEqual(len(gpng.data), 5)
        self.assertEqual(gpng.data[0], {
            'name': 'yaaa',
            'left': 1,
            'right': 20
        })
        self.assertEqual(gpng.data[1], {
            'name': 'cydc',
            'left': 25,
            'right': 30
        })
        self.assertEqual(gpng.data[2], {
            'name': 'mepm',
            'left': 50,
            'right': 63
        })
        self.assertEqual(gpng.data[3], {
            'name': 'fnge',
            'left': 65,
            'right': 70
        })
        self.assertEqual(gpng.data[4], {
            'name': 'arga',
            'left': 75,
            'right': 100
        })

    def test_circular_gene_proximity_network_generation(self):
        gpng = file_parser.GPNGenerator('tests/data/gene_proximity_network.txt')
        self.assertEqual(gpng.max_right, 100)

        graph = gpng.generate_gene_proximity_network_circular(10)
        self.assertIsInstance(graph, nx.MultiDiGraph)

        self.assertEqual(len(graph.nodes()), 5)
        self.assertEqual(set(graph.nodes()), set(['arga', 'cydc', 'fnge', 'mepm', 'yaaa']))
        self.assertEqual(set(graph.edges()), set([
            ('yaaa', 'cydc'), ('cydc', 'yaaa'),
            ('fnge', 'mepm'), ('mepm', 'fnge'),
            ('arga', 'yaaa'), ('yaaa', 'arga'),
            ('arga', 'fnge'), ('fnge', 'arga'),
            ('arga', 'fnge'), ('fnge', 'arga')
        ]))

    def test_two_strand_gene_proximity_network_generation(self):
        gpng = file_parser.GPNGenerator('tests/data/gene_proximity_network.txt')
        self.assertEqual(gpng.max_right, 100)

        graph = gpng.generate_gene_proximity_network_two_strands(10, 72)
        self.assertIsInstance(graph, nx.MultiDiGraph)

        self.assertEqual(len(graph.nodes()), 5)
        self.assertEqual(set(graph.nodes()), set(['arga', 'cydc', 'fnge', 'mepm', 'yaaa']))
        self.assertEqual(set(graph.edges()), set([
            ('yaaa', 'arga'), ('arga', 'yaaa'),
            ('fnge', 'mepm'), ('mepm', 'fnge')
        ]))

    def test_gpn_generator_terminus_computation(self):
        gpng = file_parser.GPNGenerator('tests/data/gene_proximity_network.txt')
        gpng.max_right = 9

        self.assertEqual(gpng._get_terminus(1), 5)
        self.assertEqual(gpng._get_terminus(2), 6)
        self.assertEqual(gpng._get_terminus(3), 7)
        self.assertEqual(gpng._get_terminus(4), 8)
        self.assertEqual(gpng._get_terminus(5), 9)
        self.assertEqual(gpng._get_terminus(6), 1)
        self.assertEqual(gpng._get_terminus(7), 2)
        self.assertEqual(gpng._get_terminus(8), 3)
        self.assertEqual(gpng._get_terminus(9), 4)

    def test_augmented_adjacency_matrix_generation(self):
        network_file = 'tests/data/trn_network.txt'
        aug_adja = file_parser.get_advanced_adjacency_matrix(network_file)

        self.assertEqual(aug_adja.shape, (3, 3))
        npt.assert_allclose(aug_adja, np.array([[0,1,1], [-1,0,0], [0,0,-1]]))

    def test_rnaseq_parser(self):
        exp = file_parser.parse_rnaseq('tests/data/rnaseq.count')

        self.assertEqual(len(exp.data['RNAseq']), 3)
        self.assertEqual(exp.data['RNAseq']['aaea'], 13)
        self.assertEqual(exp.data['RNAseq']['arga'], 100)
        self.assertEqual(exp.data['RNAseq']['zuzu'], 42)
        self.assertEqual(exp.get_genes(), ['aaea', 'arga', 'zuzu'])
        self.assertEqual(exp.filename, 'rnaseq.count')
