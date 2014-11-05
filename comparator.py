import sys

import numpy.linalg as npl

import utils, models


# create graph
graph = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
#graph = utils.GraphGenerator.get_random_graph(node_num=200)

# get concentrations
#concentrations = graph.io.load_concentrations('../data/concentrations/GDS3597.soft')
concentrations = graph.io.load_averaged_concentrations('../data/concentrations/')
#concentrations = graph.system.simulate(models.LinearModel)[-1,:]
#concentrations = graph.system.simulate(models.NonlinearModel)[-1,:]
#concentrations = graph.system.simulate(models.BooleanModel)[-1,:]

concentrations /= npl.norm(concentrations)

# compute pf-ev
perron_frobenius = graph.math.get_perron_frobenius()
if perron_frobenius is None:
    print('Could not determine pf ev, aborting...')
    sys.exit(1)

# plot results
utils.Plotter.plot_loglog(
    concentrations, perron_frobenius,
    'Real-life comparisons', 'Gene Concentration', 'Perron-Frobenius Eigenvector'
)
