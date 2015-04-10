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
    dic = utils.CacheHandler.store_plot_data(title, func, *args, model=model)
    func(dic)


##################
# Real-life data #
##################

def real_life_single():
    def process_file(g, pf_tmp, pr_tmp, file):
        try:
            exp = g.io.load_concentrations('../data/concentrations/%s' % file)
        except errors.InvalidGDSFormatError as e:
            print('Could not process "%s" (%s)' % (file, e))
            return

        for col, conc in exp.get_data():
            pf = exp.trim_input(pf_tmp, g, col)
            pr = exp.trim_input(pr_tmp, g, col)

            spec = '%s, %s' % (os.path.splitext(exp.filename)[0], col)
            present(
                'Real-Life Data of %s against PF' % spec, plotter.Plotter.loglog,
                'gene concentration', conc,
                'perron-frobenius eigenvector', pf
            )
            present(
                'Real-Life Data of %s against pagerank' % spec, plotter.Plotter.loglog,
                'gene concentration', conc,
                'pagerank', pr
            )
            present(
                'Histogram of Real-Life Data for %s' % spec, plotter.Plotter.plot_histogram,
                'gene concentration', 'count', conc
            )

    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt', '../data/architecture/genome.txt', 50000)
    pf_tmp = g.math.get_perron_frobenius()
    pr_tmp = g.math.get_pagerank()

    for f in os.listdir('../data/concentrations/'):
        process_file(g, pf_tmp, pr_tmp, f)

def real_life_all():
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt', '../data/architecture/genome.txt', 50000)
    pf_tmp = g.math.get_perron_frobenius()
    pr_tmp = g.math.get_pagerank()

    pf_vec = []
    pr_vec = []
    conc_vec = []
    for fname in os.listdir('../data/concentrations/'):
        try:
            exp = g.io.load_concentrations('../data/concentrations/%s' % fname)
        except errors.InvalidGDSFormatError as e:
            print('Could not process "%s" (%s)' % (fname, e))
            continue

        for col, conc in exp.get_data():
            pf = exp.trim_input(pf_tmp, g, col)
            pr = exp.trim_input(pr_tmp, g, col)

            pf_vec.extend(pf)
            pr_vec.extend(pr)
            conc_vec.extend(conc)

    present(
        'Real-Life data PF (all)', plotter.Plotter.loglog,
        'gene concentration', conc_vec,
        'perron-frobenius eigenvector', pf_vec
    )

    present(
        'Real-Life data pagerank (all)', plotter.Plotter.loglog,
        'gene concentration', conc_vec,
        'pagerank', pr_vec
    )

    present(
        'Histogram of Real-Life Data (all)', plotter.Plotter.plot_histogram,
        'gene concentration', 'count', conc_vec
    )

def real_life_average():
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt', '../data/architecture/genome.txt', 50000)

    exp = g.io.load_averaged_concentrations('../data/concentrations/', cache_file='averaged_data.csv')

    pf_tmp = g.math.get_perron_frobenius()
    pr_tmp = g.math.get_pagerank()

    col, conc = next(exp.get_data())
    pf = exp.trim_input(pf_tmp, g, col)
    pr = exp.trim_input(pr_tmp, g, col)

    present(
        'Real-Life data PF (averaged)', plotter.Plotter.loglog,
        'averaged gene concentration', conc,
        'perron-frobenius eigenvector', pf
    )

    present(
        'Real-Life data pagerank (averaged)', plotter.Plotter.loglog,
        'averaged gene concentration', conc,
        'pagerank', pr
    )

    present(
        'Histogram of Real-Life Data (averaged)', plotter.Plotter.plot_histogram,
        'gene concentration', 'count', conc
    )

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

def investigate_base_window_influence():
    """ Investigate how the size of the base window used to generate the GPN influences the correlation between the gene expression vector and the PF
    """
    def get_stuff(base_window):
        """ Get gene expression vector and PF of GPN for given base window
        """
        graph = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt', '../data/architecture/genome.txt', base_window)
        c, used_gene_indices = graph.io.load_averaged_concentrations('../data/concentrations/')

        pf_tmp = graph.math.get_perron_frobenius()
        pf = [pf_tmp[i] for i in used_gene_indices]

        return c, pf

    correlations = []
    used_windows = []

    base_windows = range(int(5e3), int(100e3), int(1e3))
    for bwin in base_windows:
        print('Base window:', bwin)
        try:
            c, pf = get_stuff(bwin)
        except (errors.PFComputationError, errors.PowerIterationError) as e:
            print('Skipped base window size %d (%s)' % (bwin, e))
            continue
        finally:
            print()

        corr, p_val = utils.StatsHandler.correlate(c, pf)

        correlations.append(corr)
        used_windows.append(bwin)

    present(
        'Effect of base window size in GPN', plotter.Plotter.plot,
        'base window size', used_windows,
        'correlation between gene expression vector and PF', correlations
    )


##################
# Generated data #
##################

def simulate_model(Model, n=20, ae=0, ie=50, plot_jc_ev=False, info={}, **kwargs):
    g = utils.GraphGenerator.get_random_graph(n, activating_edges=ae, inhibiting_edges=ie)
    Model.info.update(info)

    sim = g.system.simulate(Model, **kwargs)
    avg_data = [np.mean(time_unit) for time_unit in sim.T]

    pf = g.math.get_perron_frobenius(remove_self_links=True)
    if plot_jc_ev: ev = g.system.used_model.math.get_jacobian_ev(avg_data)

    #show_evolution(g, sim)#, pf=pf)
    present(
        '%s with PF of A' % Model.info['name'], plotter.Plotter.plot,
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
            '%s with EV of J' % Model.info['name'], plotter.Plotter.plot,
            'gene concentration', avg_data,
            'jacobian eigenvector of highest eigenvalue', ev,
            model=Model
        )

def investigate_active_edge_count_influence(Model, node_num=100, edge_num=50, repeats=10):
    """ Compute correlation between pf and gene concentrations for varying numbers of activating links in network.
        repeats specifies how many runs are average for one specific number of activating links
    """
    g = utils.GraphGenerator.get_random_graph(node_num, activating_edges=0, inhibiting_edges=edge_num)

    dats = {
        'PF': {
            'get': lambda graph: graph.math.get_perron_frobenius(remove_self_links=True)
        },
        'Pagerank': {
            'get': lambda graph: graph.math.get_pagerank()
        },
        'Node degree': {
            'get': lambda graph: graph.math.get_degree_distribution()
        }
    }

    for k in dats:
        dats[k]['correlations'] = []
        dats[k]['tmp'] = []
        dats[k]['val'] = 0

    for enu in range(edge_num+1):
        g = utils.GraphGenerator.get_random_graph(g, activating_edges=enu, inhibiting_edges=edge_num-enu)
        for k in dats: dats[k]['val'] = dats[k]['get'](g)

        for k in dats: dats[k]['tmp'] = []
        for i in range(repeats):
            sim = g.system.simulate(Model)
            avg_data = [np.mean(time_unit) for time_unit in sim.T]

            # do statistics
            for k in dats:
                corr, _ = utils.StatsHandler.correlate(dats[k]['val'], avg_data)
                dats[k]['tmp'].append(corr)

        for k in dats: dats[k]['correlations'].append(dats[k]['tmp'])

    for k in dats:
        present(
            '%s correlation development for increasing number of activating links (%s)' % (k, Model.info['name']), plotter.Plotter.errorbar_plot,
            'number of activating links', list(range(edge_num+1)),
            'correlation coefficient', dats[k]['correlations']
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

    #simulate_model(models.MultiplicatorModel, runs=20)
    #simulate_model(models.BooleanModel)
    #simulate_model(models.LinearModel, plot_jc_ev=True, runs=20)
    #simulate_model(models.NonlinearModel, plot_jc_ev=True, runs=20)

    #analysis(utils.GraphGenerator.get_er_graph(100, 0.3), models.MultiplicatorModel)
    investigate_active_edge_count_influence(models.BooleanModel)

    #gene_overview()
    #investigate_base_window_influence()

    #real_life_average()
    #real_life_all()
    #real_life_single()
