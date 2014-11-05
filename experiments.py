import sys

import numpy.linalg as npl

import utils, models, plotter


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
    plotter.Plotter.loglog(
        concentrations, perron_frobenius,
        title, xlabel, ylabel
    )

def analysis(graph, Model):
    """ Examines different concentration components vs pf-ev
    """
    # generate data
    pf = graph.math.get_perron_frobenius()
    data = graph.system.simulate(Model)

    # get individual points
    fpc = data[-1,:]
    pc = data[5,:]
    fpc_pc = fpc - pc
    pc_pc = pc - data[9,:]

    # plot result
    plotter.Plotter.multi_loglog(
        'Analysis of %s' % Model.name,
        'perron-frobenius eigenvector', [
        {
            'x': pf,
            'y': fpc,
            'ylabel': 'FP [c]'
        }, {
            'x': pf,
            'y': pc,
            'ylabel': 'P [c]'
        }, {
            'x': pf,
            'y': fpc_pc,
            'ylabel': 'FP-P [c]'
        }, {
            'x': pf,
            'y': pc_pc,
            'ylabel': 'P-P [c]'
        }
    ])


##################
# Real-life data #
##################

def real_life_single():
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
    c = g.io.load_concentrations('../data/concentrations/GDS3597.soft')

    loglog(
        g, c,
        'Real-Life Data', 'gene concentration', 'perron-frobenius eigenvector'
    )

def real_life_average():
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
    c = g.io.load_averaged_concentrations('../data/concentrations/')

    loglog(
        g, c,
        'Real-Life Data (averaged)', 'averaged gene concentration', 'perron-frobenius eigenvector'
    )


##################
# Generated data #
##################

def boolean_model(n=100, e=0.3):
    g = utils.GraphGenerator.get_random_graph(node_num=n, edge_prob=e)
    c = g.system.simulate(models.BooleanModel)[-1,:]

    loglog(
        g, c,
        'Boolean Model', 'gene concentration', 'perron-frobenius eigenvector'
    )

def linear_model(n=100, e=0.3):
    g = utils.GraphGenerator.get_random_graph(node_num=n, edge_prob=e)
    c = g.system.simulate(models.LinearModel)[-1,:]

    loglog(
        g, c,
        'Linear Model', 'gene concentration', 'perron-frobenius eigenvector'
    )

def nonlinear_model(n=100, e=0.3):
    g = utils.GraphGenerator.get_random_graph(node_num=n, edge_prob=e)
    c = g.system.simulate(models.NonlinearModel)[-1,:]

    loglog(
        g, c,
        'Nonlinear Model', 'gene concentration', 'perron-frobenius eigenvector'
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
    nonlinear_model()

    #analysis(utils.GraphGenerator.get_random_graph(100, 0.3), models.MultiplicatorModel)
