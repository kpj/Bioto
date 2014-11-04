import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

import numpy as np
import numpy.linalg as npl

import networkx as nx

import utils, parser, data_generator


# create graph
g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
#g = utils.GraphGenerator.get_random_graph(node_num=20)

# compute data
perron_frobenius = g.get_perron_frobenius()

# get concentrations
#concentrations = utils.DataHandler.load_concentrations(g, '../data/concentrations/GDS3597_full.soft')
concentrations = utils.DataHandler.load_averaged_concentrations(g, '../data/concentrations/')
#concentrations = data_generator.NonlinearModel(g).generate()[-1,:]
#concentrations = data_generator.BooleanModel(g).generate()[-1,:]

concentrations /= npl.norm(concentrations)

# plot results
utils.Plotter.plot_loglog(
    concentrations, perron_frobenius,
    'Real-life comparisons', 'Gene Concentration', 'Perron-Frobenius Eigenvector'
)
