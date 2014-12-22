import sys, os, random

import numpy as np
import numpy.linalg as npl

import utils, models, plotter


####################
# Helper functions #
####################

def present(title, func, *args):
    """ Save and (if needed) plot data
    """
    dic = utils.CacheHandler.store_plot_data(title, func, *args) # store data for later use
    func(dic) # also plot if wanted

##################
# Real-life data #
##################

def real_life_single(file):
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
    c = g.io.load_concentrations('../data/concentrations/%s' % file)

    pf = g.math.get_perron_frobenius()
    corr, p_val, = utils.StatsHandler.correlate(c, pf)

    present(
        'Real-Life Data of %s' % file, plotter.Plotter.loglog,
        'gene concentration', c,
        'perron-frobenius eigenvector', pf
    )

def real_life_average():
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
    c = g.io.load_averaged_concentrations('../data/concentrations/')

    pf = g.math.get_perron_frobenius()
    corr, p_val = utils.StatsHandler.correlate(c, pf)

    present(
        'Real-Life Data (averaged)', plotter.Plotter.loglog,
        'averaged gene concentration', c,
        'perron-frobenius eigenvector', pf
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
    present(
        '%s with PF of A' % Model.name, plotter.Plotter.loglog,
        'gene concentration', sim[-1,:],
        'perron-frobenius eigenvector', pf
    )
    present(
        '%s with PF of A (delta)' % Model.name, plotter.Plotter.loglog,
        'difference in gene concentration', sim[-1,:]-sim[-10,:],
        'perron-frobenius eigenvector', pf
    )
    if plot_jc_ev:
        present(
            '%s with EV of J' % Model.name, plotter.Plotter.loglog,
            'gene concentration', sim[-1,:],
            'jacobian eigenvector of highest eigenvalue', ev
        )

def investigate_active_edge_count_influence(Model, n=100, e=0.3, repeats=5):
    """ Compute correlation between pf and gene concentrations for varying numbers of activating links in network.
        repeats specifies how many runs are average for one specific number of activating links
    """
    g = utils.GraphGenerator.get_random_graph(node_num=n, edge_prob=e)
    pf = g.math.get_perron_frobenius(remove_self_links=True)

    aug_adja_m = np.copy(g.adja_m).T
    one_num = int(aug_adja_m.sum())
    x_range = list(range(one_num+1))

    # save indices of ones (for easier access later on)
    indices = []
    for i in range(aug_adja_m.shape[0]):
        for j in range(aug_adja_m.shape[1]):
            if aug_adja_m[i,j] == 1:
                indices.append((i,j))

    # make audgmented adja_m inhibitory only
    aug_adja_m[aug_adja_m == 1] = -1

    correlations = []
    while True:
        tmp = []
        for i in range(repeats):
            # extract some entry from simulation
            sim = g.system.simulate(Model, 5, aug_adja=aug_adja_m)
            exprs = sim[-1,:]

            # do statistics
            corr, p_val = utils.StatsHandler.correlate(pf, exprs)
            tmp.append(corr)
        correlations.append(tmp)

        if one_num == 0:
            break

        # randomly activate some edge
        ind = random.choice(indices)
        indices.remove(ind)
        aug_adja_m[ind] = 1

        one_num -= 1

    present(
        'Correlation development for increasing number of activating links', plotter.Plotter.errorbar_plot,
        'number of activating links', x_range,
        'correlation coefficient', correlations
    )

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

    present(
        'System Evolution of %s' % graph.system.used_model.name, plotter.Plotter.multi_plot,
        'time', 'simulated gene expression level',
        data
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
    present(
        'Analysis of %s' % Model.name, plotter.Plotter.multi_loglog,
        'perron-frobenius eigenvector',
        data
    )


##################
# Command Center #
##################

if __name__ == '__main__':
    plotter.Plotter.show_plots = False

    simulate_model(models.MultiplicatorModel)
    simulate_model(models.BooleanModel)
    simulate_model(models.LinearModel, plot_jc_ev=True)
    simulate_model(models.NonlinearModel, plot_jc_ev=True)

    analysis(utils.GraphGenerator.get_random_graph(100, 0.3), models.MultiplicatorModel)

    investigate_active_edge_count_influence(models.BooleanModel, n=10, repeats=2)

    real_life_average()
    for f in os.listdir('../data/concentrations/'): real_life_single(f)
