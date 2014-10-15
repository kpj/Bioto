import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

import numpy as np

import utils


# config
node_num = 10
edge_prob = 0.3
runs = 11

# init
graph_handler = utils.GraphHandler(utils.GraphGenerator.get_random_graph(node_num, edge_prob))

# generate data
data = graph_handler.simulate(runs)

# get perron frobenius eigenvector/page rank
perron_frobenius = graph_handler.get_perron_frobenius()
page_rank = graph_handler.get_pagerank()
degree_distribution = graph_handler.get_degree_distribution()

# plot it
utils.Plotter.present_graph(data, perron_frobenius, page_rank, degree_distribution)
