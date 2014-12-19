import sys, os

import numpy as np
import numpy.linalg as npl

import utils, models, plotter


####################
# Helper functions #
####################

def show_evolution(graph, sim, t_window, genes=range(5), pf=None):
    """ Plots evolution of individual genes over time interval
    """
    data = []
    for ge in genes:
        evolution_data = sim.T[ge]
        evolen = len(evolution_data)

        gene_evolution = {
            'x': range(evolen),
            'y': evolution_data,
            'label': 'gene %i' % ge
        }
        data.append(gene_evolution)

        if not pf is None:
            pf_ev = {
                'x': range(evolen),
                'y': [pf[ge]]*evolen,
                'label': 'pf comp for gene %i' % ge
            }
            data.append(pf_ev)

    plotter.Plotter.multi_plot(
        data,
        'System Evolution of %s' % graph.system.used_model.name,
        'time', 'simulated gene expression level'
    )

def analysis(graph, Model, runs=10):
    """ Examines different concentration components vs pf-ev
    """
    # generate data
    pf = graph.math.get_perron_frobenius()
    sim = graph.system.simulate(Model, runs)

    # gather data
    data = []
    for i in range(0, runs, 2):
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

    pf = g.math.get_perron_frobenius()
    corr, p_val, = utils.StatsHandler.correlate(c, pf)

    plotter.Plotter.loglog(
        c, pf,
        'Real-Life Data of %s' % file, 'gene concentration', 'perron-frobenius eigenvector'
    )

def real_life_average():
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
    c = g.io.load_averaged_concentrations('../data/concentrations/')

    pf = g.math.get_perron_frobenius()
    corr, p_val = utils.StatsHandler.correlate(c, pf)

    plotter.Plotter.loglog(
        c, pf,
        'Real-Life Data (averaged)', 'averaged gene concentration', 'perron-frobenius eigenvector'
    )


##################
# Generated data #
##################

def simulate_model(Model, n=100, e=0.3, runs=15, plot_jc_ev=False):
    g = utils.GraphGenerator.get_random_graph(node_num=n, edge_prob=e)

    sim = g.system.simulate(Model, runs)
    pf = g.math.get_perron_frobenius(remove_self_links=True)
    if plot_jc_ev: ev = g.system.used_model.math.get_jacobian_ev(sim[-1,:])

    show_evolution(g, sim, runs)#, pf=pf)
    plotter.Plotter.loglog(
        sim[-1,:], pf,
        '%s with PF of A' % Model.name, 'gene concentration', 'perron-frobenius eigenvector'
    )
    if plot_jc_ev:
        plotter.Plotter.plot(
            sim[-1,:], ev,
            '%s with EV of J' % Model.name, 'gene concentration', 'jacobian eigenvector of highest eigenvalue'
        )


##################
# Command Center #
##################

if __name__ == '__main__':
    plotter.Plotter.show_plots = False

    #for f in os.listdir('../data/concentrations/'): real_life_single(f)
    #real_life_average()

    #simulate_model(models.MultiplicatorModel)
    #simulate_model(models.BooleanModel, n=50, e=0.9)
    #simulate_model(models.LinearModel, plot_jc_ev=True)
    #simulate_model(models.NonlinearModel, plot_jc_ev=True)

    #analysis(utils.GraphGenerator.get_random_graph(100, 0.3), models.BooleanModel)
