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
    #concentrations /= npl.norm(concentrations)

    # plot data
    plotter.Plotter.loglog(
        concentrations, perron_frobenius,
        title, xlabel, ylabel
    )

def show_evolution(graph, sim, t_window, genes=range(5)):
    """ Plots evolution of individual genes over time interval
    """
    pf = graph.math.get_perron_frobenius()
    if pf is None:
        print('Could not determine pf ev, aborting...')
        sys.exit(1)

    data = []
    for ge in genes:
        evolution_data = sim.T[ge]
        evolen = len(evolution_data)

        gene_evolution = {
            'x': range(evolen),
            'y': evolution_data,
            'label': 'gene %i' % ge
        }
        pf_ev = {
            'x': range(evolen),
            'y': [pf[ge]]*evolen,
            'label': 'pf comp for gene %i' % ge
        }

        data.append(gene_evolution)
        data.append(pf_ev)

    plotter.Plotter.multi_plot('System Evolution of %s' % graph.system.used_model.name, data)

def analysis(graph, Model, runs=10):
    """ Examines different concentration components vs pf-ev
    """
    # generate data
    pf = graph.math.get_perron_frobenius()
    sim = graph.system.simulate(Model, runs)

    # gather data
    data = []
    for i in range(runs):
        cur = {
            'x': pf,
            'y': sim[i,:],
            'ylabel': 't: %i' % i
        }

        data.append(cur)

    # plot result
    plotter.Plotter.multi_loglog('Analysis of %s' % Model.name, 'perron-frobenius eigenvector', data)


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

def simulate_model(Model, n=100, e=0.3, runs=20):
    g = utils.GraphGenerator.get_random_graph(node_num=n, edge_prob=e)
    sim = g.system.simulate(Model, runs)

    show_evolution(g, sim, runs)
    loglog(
        g, sim[-1,:],
        Model.name, 'gene concentration', 'perron-frobenius eigenvector'
    )


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

    analysis(utils.GraphGenerator.get_random_graph(100, 0.3), models.BooleanModel)
