import sys, random, collections
import os, os.path

import numpy as np
import numpy.linalg as npl

import utils, models, plotter


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
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
    c, used_gene_indices = g.io.load_concentrations('../data/concentrations/%s' % file)

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
    g = utils.GraphGenerator.get_random_graph(node_num=n, activating_edges=ae, inhibiting_edges=ie)
    #g.io.visualize('random.png', use_R=False); sys.exit()
    Model.info.update(info)

    sim = g.system.simulate(Model)
    bm_data = [np.mean(time_unit) for time_unit in sim.T]

    pf = g.math.get_perron_frobenius(remove_self_links=True)
    if plot_jc_ev: ev = g.system.used_model.math.get_jacobian_ev(bm_data)

    #show_evolution(g, sim)#, pf=pf)
    present(
        '%s with PF of A' % Model.info['name'], plotter.Plotter.loglog,
        'gene concentration', bm_data,
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
            'gene concentration', bm_data,
            'jacobian eigenvector of highest eigenvalue', ev,
            model=Model
        )

def investigate_active_edge_count_influence(Model, n=100, e=0.3, repeats=5):
    """ Compute correlation between pf and gene concentrations for varying numbers of activating links in network.
        repeats specifies how many runs are average for one specific number of activating links
    """
    g = utils.GraphGenerator.get_er_graph(node_num=n, edge_prob=e)
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

    # make augmented adja_m inhibitory only
    aug_adja_m[aug_adja_m == 1] = -1

    correlations = []
    while True:
        tmp_rep = []
        for i in range(repeats):
            if len(indices) == 0:
                indices.append((-1,-1))

            tmp_ind = []
            for ind in indices:
                tmp_aam = np.copy(aug_adja_m)
                if ind != (-1,-1):
                    tmp_aam[ind] = 1

                # extract some entry from simulation
                sim = g.system.simulate(Model, aug_adja=tmp_aam)
                exprs = sim[-1,:]

                # do statistics
                corr, p_val = utils.StatsHandler.correlate(pf, exprs)
                tmp_ind.append(corr)

            tmp_rep.extend(tmp_ind)
        correlations.append(tmp_rep)

        if one_num == 0:
            break

        # randomly activate some edge for next stage
        ind = random.choice(indices)
        indices.remove(ind)
        aug_adja_m[ind] = 1

        one_num -= 1

    present(
        'Correlation development for increasing number of activating links (%s)' % Model.info['name'], plotter.Plotter.errorbar_plot,
        'number of activating links', x_range,
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
    plotter.Plotter.show_plots = True

    #simulate_model(models.MultiplicatorModel)
    #simulate_model(models.BooleanModel)
    #simulate_model(models.LinearModel, plot_jc_ev=True)
    #simulate_model(models.NonlinearModel, plot_jc_ev=True)

    #analysis(utils.GraphGenerator.get_er_graph(100, 0.3), models.MultiplicatorModel)
    #investigate_active_edge_count_influence(models.MultiplicatorModel, n=10, repeats=2)

    #gene_overview()

    real_life_average()
    #for f in os.listdir('../data/concentrations/'): real_life_single(f)
