import sys, random, collections
import os, os.path

import numpy as np
import numpy.linalg as npl

import utils, models, plotter, errors


####################
# Helper functions #
####################

def present(title, func, *args, model=None):
    """ Save and (if needed) plot data
    """
    dic = utils.CacheHandler.store_plot_data(title, func, *args, model=model) # store data for later use
    func(dic) # also plot if wanted


##################
# Real-life data #
##################

def real_life_single(file):
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt', '../data/architecture/genome.txt', 50000)

    try:
        c, used_gene_indices = g.io.load_concentrations('../data/concentrations/%s' % file)
    except errors.InvalidGDSFormatError as e:
        print('Could not process "%s" (%s)' % (file, e))
        return

    pf_tmp = g.math.get_perron_frobenius()
    pf = [pf_tmp[i] for i in used_gene_indices]
    corr, p_val, = utils.StatsHandler.correlate(c, pf)

    present(
        'Real-Life Data of %s' % file, plotter.Plotter.loglog,
        'gene concentration', c,
        'perron-frobenius eigenvector', pf
    )

def real_life_average():
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt', '../data/architecture/genome.txt', 50000)

    c, used_gene_indices = g.io.load_averaged_concentrations('../data/concentrations/')

    pf_tmp = g.math.get_perron_frobenius()
    pf = [pf_tmp[i] for i in used_gene_indices]
    corr, p_val = utils.StatsHandler.correlate(c, pf)
    present(
        'Real-Life Data (averaged)', plotter.Plotter.loglog,
        'averaged gene concentration', c,
        'perron-frobenius eigenvector', pf
    )

    """pr_tmp = g.math.get_pagerank()
    pr = [pr_tmp[i] for i in used_gene_indices]
    corr, p_val = utils.StatsHandler.correlate(c, pr)
    present(
        'Real-Life Data (averaged)', plotter.Plotter.loglog,
        'averaged gene concentration', c,
        'page rank', pr
    )"""

def gene_overview(density_plot=True):
    graph = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')

    names = list(graph)
    dire = '../data/concentrations/'
    files = os.listdir(dire)

    # gather data
    data = collections.defaultdict(list)
    for f in files:
        c_tmp, used_gene_indices = graph.io.load_concentrations(os.path.join(dire, f))
        c = [c_tmp[i] for i in used_gene_indices]

        for i, name in enumerate(names):
            if i == len(used_gene_indices): break
            data[name].append(c[i])

    # plot data
    if density_plot:
        tmp = []
        for gene in names:
            cur = data[gene]
            tmp.append({
                'name': gene,
                'data': cur
            })

        plotter.Plotter.multi_density_plot(tmp, 'Overview over gene expression level distributions', 'gene expression level', 'number of occurences')
    else:
        num = len(files)
        plots = []
        for i in range(num):
            tmp = []
            for name in names:
                tmp.append(data[name][i])

            e = {
                'x': range(len(names)),
                'y': tmp,
                'label': files[i]
            }
            plots.append(e)

        present(
            'Gene expression overview in GDS files', plotter.Plotter.multi_plot,
            'gene', 'gene expression level',
            plots
        )


##################
# Generated data #
##################

def simulate_model(Model, n=20, ae=0, ie=50, plot_jc_ev=False, info={}):
    g = utils.GraphGenerator.get_random_graph(n, activating_edges=ae, inhibiting_edges=ie)
    #g.io.visualize('random.png', use_R=False); sys.exit()
    Model.info.update(info)

    sim = g.system.simulate(Model)
    avg_data = [np.mean(time_unit) for time_unit in sim.T]

    pf = g.math.get_perron_frobenius(remove_self_links=True)
    if plot_jc_ev: ev = g.system.used_model.math.get_jacobian_ev(avg_data)

    #show_evolution(g, sim)#, pf=pf)
    present(
        '%s with PF of A' % Model.info['name'], plotter.Plotter.loglog,
        'gene concentration', avg_data,
        'perron-frobenius eigenvector', pf,
        model=Model
    )
    #present(
    #    '%s with PF of A (delta)' % Model.info['name'], plotter.Plotter.loglog,
    #    'difference in gene concentration', sim[-1,:]-sim[-10,:],
    #    'perron-frobenius eigenvector', pf,
    #    model=Model
    #)
    if plot_jc_ev:
        present(
            '%s with EV of J' % Model.info['name'], plotter.Plotter.loglog,
            'gene concentration', avg_data,
            'jacobian eigenvector of highest eigenvalue', ev,
            model=Model
        )

def investigate_active_edge_count_influence(Model, node_num=100, edge_num=50, repeats=10):
    """ Compute correlation between pf and gene concentrations for varying numbers of activating links in network.
        repeats specifies how many runs are average for one specific number of activating links
    """
    g = utils.GraphGenerator.get_random_graph(node_num, activating_edges=0, inhibiting_edges=edge_num)
    pf = g.math.get_perron_frobenius(remove_self_links=True)

    correlations = []
    for enu in range(edge_num+1):
        g = utils.GraphGenerator.get_random_graph(g, activating_edges=enu, inhibiting_edges=edge_num-enu)

        tmp_corr = []
        for i in range(repeats):
            sim = g.system.simulate(Model)
            avg_data = [np.mean(time_unit) for time_unit in sim.T]

            # do statistics
            corr, p_val = utils.StatsHandler.correlate(pf, avg_data)
            tmp_corr.append(corr)

        correlations.append(tmp_corr)

    present(
        'Correlation development for increasing number of activating links (%s)' % Model.info['name'], plotter.Plotter.errorbar_plot,
        'number of activating links', list(range(edge_num+1)),
        'correlation coefficient', correlations
    )

def show_evolution(graph, sim, genes=range(5), pf=None):
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
        'System Evolution of %s' % graph.system.used_model.info['name'], plotter.Plotter.multi_plot,
        'time', 'simulated gene expression level',
        data
    )

def analysis(graph, Model, runs=10):
    """ Examines different concentration components vs pf-ev
    """
    # generate data
    pf = graph.math.get_perron_frobenius()
    sim = graph.system.simulate(Model)

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
        'Analysis of %s' % Model.info['name'], plotter.Plotter.multi_loglog,
        'perron-frobenius eigenvector',
        data,
        model=Model
    )


##################
# Command Center #
##################

if __name__ == '__main__':
    plotter.Plotter.show_plots = False

    #simulate_model(models.MultiplicatorModel)
    #simulate_model(models.BooleanModel)
    #simulate_model(models.LinearModel, plot_jc_ev=True)
    #simulate_model(models.NonlinearModel, plot_jc_ev=True)

    #analysis(utils.GraphGenerator.get_er_graph(100, 0.3), models.MultiplicatorModel)
    investigate_active_edge_count_influence(models.BooleanModel)

    #gene_overview()

    #real_life_average()
    #for f in os.listdir('../data/concentrations/'): real_life_single(f)
