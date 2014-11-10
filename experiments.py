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

def real_life_single(file):
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
    c = g.io.load_concentrations('../data/concentrations/%s' % file)

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

def simulate_model(Model, n=100, e=0.3):
    g = utils.GraphGenerator.get_random_graph(node_num=n, edge_prob=e)
    c = g.system.simulate(Model)[-1,:]

    loglog(
        g, c,
        Model.name, 'gene concentration', 'perron-frobenius eigenvector'
    )

def show_evolution(Model, n=10, e=0.3, t_window=100, genes=range(5)):
    """ Plots evolution of individual genes over time interval
    """
    g = utils.GraphGenerator.get_random_graph(node_num=n, edge_prob=e)
    sim = g.system.simulate(Model, t_window)

    pf = g.math.get_perron_frobenius()

    data = []
    for ge in genes:
        gene_evolution = {
            'x': range(t_window),
            'y': sim.T[ge][:t_window], # cut off if there is too much data available
            'label': 'gene %i' % ge
        }
        pf_ev = {
            'x': range(t_window),
            'y': [pf[ge]]*t_window,
            'label': 'pf comp for gene %i' % ge
        }

        data.append(gene_evolution)
        data.append(pf_ev)

    plotter.Plotter.multi_plot('System Evolution of %s' % Model.name, data)


##################
# Command Center #
##################

if __name__ == '__main__':
    plotter.Plotter.show_plots = True

    #real_life_single('GDS3597.soft')
    #real_life_average()

    #simulate_model(models.MultiplicatorModel)
    #simulate_model(models.BooleanModel)
    #simulate_model(models.LinearModel)
    #simulate_model(models.NonlinearModel)

    #show_evolution(models.MultiplicatorModel, t_window=20)
    #show_evolution(models.BooleanModel)
    #show_evolution(models.LinearModel)
    show_evolution(models.NonlinearModel)

    #analysis(utils.GraphGenerator.get_random_graph(100, 0.3), models.MultiplicatorModel)
