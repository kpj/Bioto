import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

import numpy as np
import numpy.linalg as npl

import networkx as nx

import utils, parser, data_generator


# create graph
g = utils.GraphHandler(
    utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
)
#g.visualize('regnet.png')
#g.dump_adjacency_matrix('foo.txt')
#g.dump_node_names('names.txt')

# compute data
perron_frobenius = g.get_perron_frobenius()

# get concentrations
#concentrations = utils.DataHandler.load_concentrations(g, '../data/concentrations/GDS3597_full.soft')
#concentrations = utils.DataHandler.load_averaged_concentrations(g, '../data/concentrations/')
#concentrations = data_generator.NonlinearModel('../data/architecture/network_tf_gene.txt').generate()[-1,:]
concentrations = data_generator.BooleanModel(node_num=len(g)).generate()[-1,:]

concentrations /= npl.norm(concentrations)

# plot results
utils.Plotter.plot_loglog(
    concentrations, perron_frobenius,
    'Real-life comparisons', 'Gene Concentrations', 'Perron-Frobenius Eigenvector'
)
