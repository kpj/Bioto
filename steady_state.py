import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

import numpy as np

import utils, models


# config
node_num = 50
edge_prob = 0.3
runs = 11

# init
graph_handler = utils.GraphGenerator.get_random_graph(node_num, edge_prob)

# generate data
data = graph_handler.system.simulate(models.MultiplicatorModel, runs)

# get perron frobenius eigenvector/page rank
perron_frobenius = graph_handler.math.get_perron_frobenius()
page_rank = graph_handler.math.get_pagerank()
degree_distribution = graph_handler.math.get_degree_distribution()

# compute some correlations
r, min_b, max_b = utils.StatsHandler.correlate(perron_frobenius, data[-1])
print("pf/conc corr", r)
print("Bands:", min_b, max_b)

# plot it
utils.Plotter.present_graph(data, perron_frobenius, page_rank, degree_distribution)