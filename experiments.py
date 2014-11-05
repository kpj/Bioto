import sys

import numpy.linalg as npl

import utils, models


####################
# Helper functions #
####################

def loglog(graph, concentrations, title, xlabel, ylabel):
    # get data
    perron_frobenius = graph.math.get_perron_frobenius()
    if perron_frobenius is None:
        print('Could not determine pf ev, aborting...')
        sys.exit(1)

    # normalize data
    concentrations /= npl.norm(concentrations)

    # plot data
    utils.Plotter.plot_loglog(
        concentrations, perron_frobenius,
        title, xlabel, ylabel
    )


##################
# Real-life data #
##################

def real_life_single():
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
    c = g.io.load_concentrations('../data/concentrations/GDS3597.soft')

    loglog(
        g, c,
        'Real-Life Data', 'gene concentration', 'perron-forbenius eigenvector'
    )

def real_life_average():
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
    c = g.io.load_averaged_concentrations('../data/concentrations/')

    loglog(
        g, c,
        'Real-Life Data', 'averaged gene concentration', 'perron-forbenius eigenvector'
    )


##################
# Generated data #
##################

def boolean_model(n=100, e=0.3):
    g = utils.GraphGenerator.get_random_graph(node_num=n, edge_prob=e)
    c = g.system.simulate(models.BooleanModel)[-1,:]

    loglog(
        g, c,
        'Boolean Model', 'gene concentration', 'perron-forbenius eigenvector'
    )

def linear_model(n=100, e=0.3):
    g = utils.GraphGenerator.get_random_graph(node_num=n, edge_prob=e)
    c = g.system.simulate(models.LinearModel)[-1,:]

    loglog(
        g, c,
        'Linear Model', 'gene concentration', 'perron-forbenius eigenvector'
    )

def nonlinear_model(n=100, e=0.3):
    g = utils.GraphGenerator.get_random_graph(node_num=n, edge_prob=e)
    c = g.system.simulate(models.NonlinearModel)[-1,:]

    loglog(
        g, c,
        'Nonlinear Model', 'gene concentration', 'perron-forbenius eigenvector'
    )


##################
# Command Center #
##################

if __name__ == '__main__':
    pass

    #real_life_single()
    #real_life_average()

    #boolean_model()
    #linear_model()
    #nonlinear_model()
