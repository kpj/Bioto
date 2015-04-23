import sys, random, collections
import os, os.path

import numpy as np
import numpy.linalg as npl

from progressbar import ProgressBar

import utils, models, plotter, errors, logger


####################
# Helper functions #
####################

def present(title, func, *args, model=None, **kwargs):
    """ Save and (if needed) plot data
    """
    dic = utils.CacheHandler.store_plot_data(title, func, *args, model=model, **kwargs)

    plotter.Plotter.preprocess(**kwargs)
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
        'perron-frobenius eigenvector', pf_vec,
        plt_args={'alpha': 0.02}
    )

    present(
        'Real-Life data pagerank (all)', plotter.Plotter.loglog,
        'gene concentration', conc_vec,
        'pagerank', pr_vec,
        plt_args={'alpha': 0.02}
    )

    present(
        'Histogram of Real-Life Data (all)', plotter.Plotter.plot_histogram,
        'gene concentration', 'count', conc_vec
    )

def real_life_average():
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt', '../data/architecture/genome.txt', 50000)

    exp = g.io.load_averaged_concentrations('../data/concentrations/')

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

def real_life_rnaseq():
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
    exp = g.io.load_concentrations('rnaseq_pipeline/results/SRR933989_mapped.count')

    pf_tmp = g.math.get_perron_frobenius()
    pr_tmp = g.math.get_pagerank()

    col, conc = next(exp.get_data())
    pf = exp.trim_input(pf_tmp, g, col)
    pr = exp.trim_input(pr_tmp, g, col)

    present(
        'RNAseq data vs PF', plotter.Plotter.loglog,
        'averaged gene concentration', conc,
        'perron-frobenius eigenvector', pf
    )

    present(
        'RNAseq data vs pagerank', plotter.Plotter.loglog,
        'averaged gene concentration', conc,
        'pagerank', pr
    )

    present(
        'Histogram of RNAseq Data', plotter.Plotter.plot_histogram,
        'gene concentration', 'count', conc
    )

def rnaseq_vs_microarray(rnaseq='SRR933989', gds='GDS2578', gsm='GSM99092'):
    """ Plot RNAseq data against microarray data

        The default microarray data/col choice has a histogram which is not too skewed
    """
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt', '../data/architecture/genome.txt', 50000)

    rna_exp = g.io.load_concentrations('rnaseq_pipeline/results/%s_mapped.count' % rnaseq)
    ma_exp = g.io.load_concentrations('../data/concentrations/%s.soft' % gds)

    _, rna_conc = next(rna_exp.get_data())
    _, ma_conc = next(ma_exp.get_data(cols=[gsm]))

    present(
        'Histogram for RNAseq data from %s' % rnaseq, plotter.Plotter.plot_histogram,
        'gene concentration', 'count', rna_conc
    )

    present(
        'Histogram for Microarray data from %s, %s' % (gds, gsm), plotter.Plotter.plot_histogram,
        'gene concentration', 'count', ma_conc
    )

    rsc, mac = utils.combine_gds_parse_results(rna_exp, ma_exp, ('RNAseq', 'GSM99092'))

    present(
        'RNAseq vs Microarray data', plotter.Plotter.loglog,
        'RNAseq data', rsc,
        'microarray data', mac
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

def investigate_origin_of_replication_influence():
    """ Use the two-stranded GPN with varying origin in order to investigate the importance of particular origin position
    """
    # actual origin between 130 and 370 (http://www.metalife.com/Genbank/147023)
    possible_origins = range(1, 4641628, 10000) # 4641628 is rightmost gene end (yjtD)

    # compute average gene concentrations once in advance
    g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt', '../data/architecture/genome.txt', 50000)
    exp = g.io.load_averaged_concentrations('../data/concentrations/')
    col, conc = next(exp.get_data())

    pbar = ProgressBar(maxval=len(possible_origins))
    pbar.start()

    pf_corrs = []
    for i, orig in enumerate(possible_origins):
        g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt', '../data/architecture/genome.txt', 50000, origin=orig)

        pf_tmp = g.math.get_perron_frobenius()
        pf = exp.trim_input(pf_tmp, g, col)

        corr, _ = utils.StatsHandler.correlate(pf, conc)
        pf_corrs.append(corr)

        pbar.update(i)
    pbar.finish()

    present(
        'Effect of origin of replication', plotter.Plotter.plot,
        'origin of replication', possible_origins,
        'correlation between gene expression vector and PF', pf_corrs
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

def investigate_active_edge_count_influence_quot(Model, node_num=20, edge_num=50, repeats=10, scalefree=False, indegree=True):
    """ Compute correlation between PF/NID corr. and PF/GE corr. depending on act/inh link ratio.
    """
    pbar = ProgressBar(maxval=(edge_num+1) * repeats)
    pbar.start()
    counter = 0

    pf_nid_correlations = []
    pf_ge_correlations = []
    for enu in range(edge_num+1):
        pm_tmp = []
        pg_tmp = []

        for i in range(repeats):
            # regenerate graph
            g = utils.GraphGenerator.get_random_graph(
                node_num,
                activating_edges=enu, inhibiting_edges=edge_num-enu, scalefree=scalefree
            )

            # get data
            sim = g.system.simulate(Model)
            avg_data = [np.mean(time_unit) for time_unit in sim.T]

            pf = g.math.get_perron_frobenius(remove_self_links=True)
            nd = g.math.get_degree_distribution(indegree=indegree)

            # do statistics
            pm_tmp.append(utils.StatsHandler.correlate(pf, nd)[0])
            pg_tmp.append(utils.StatsHandler.correlate(pf, avg_data)[0])

            pbar.update(counter)
            counter += 1

        pf_nid_correlations.append(pm_tmp)
        pf_ge_correlations.append(pg_tmp)
    pbar.finish()

    # compute act/inh ratio similar to TRN
    trn_ratio = 0.843879907621
    act_num = round((trn_ratio * edge_num) / (trn_ratio + 1))

    # change tags dependent on degree type
    title_tag = {True: 'in-degree', None: 'overall-degree', False: 'out-degree'}

    present(
        'Correlation development for increasing number of activating links on %s graph (%s, %s; %s)' % ('scale-free' if scalefree else 'ER', Model.info['name'], 'time norm' if models.BooleanModel.info['norm_time'] else 'gene norm', title_tag[indegree]), plotter.Plotter.errorbar_plot,
        'number of activating links', list(range(edge_num+1)),
        'correlation coefficient',
        [
            ('perron-frobenius, node %s' % title_tag[indegree], pf_nid_correlations),
            ('perron-frobenius, gene expression', pf_ge_correlations)
        ],
        axis_preprocessing={
            'axvline': ((act_num,), {'linestyle': '--', 'color': 'k'}),
            'text': ((act_num + 0.5, -0.8, 'act/inh link ratio of TRN'), {}),
            'set_ylim': (([-1, 1], {}))
        }
    )

def investigate_active_edge_count_influence_gene_expr(Model, node_num=20, edge_num=50, repeats=10):
    """ Compute correlation between pf/pagerank/node in-degree and gene concentrations for varying numbers of activating links in network.
        repeats specifies how many runs are average for one specific number of activating links
    """
    g = utils.GraphGenerator.get_random_graph(node_num, activating_edges=0, inhibiting_edges=edge_num)
    pbar = ProgressBar(maxval=(edge_num+1) * repeats)

    dats = {
        'PF': {
            'get': lambda graph: graph.math.get_perron_frobenius(remove_self_links=True)
        },
        'Pagerank': {
            'get': lambda graph: graph.math.get_pagerank()
        },
        'Node in-degree': {
            'get': lambda graph: graph.math.get_degree_distribution()
        }
    }

    for k in dats:
        dats[k]['correlations'] = []
        dats[k]['tmp'] = []
        dats[k]['val'] = 0

    pbar.start()
    counter = 0
    for enu in range(edge_num+1):
        for k in dats: dats[k]['tmp'] = []
        for i in range(repeats):
            # regenerate graph
            g = utils.GraphGenerator.get_random_graph(g, activating_edges=enu, inhibiting_edges=edge_num-enu)
            for k in dats: dats[k]['val'] = dats[k]['get'](g)

            # get data
            sim = g.system.simulate(Model)
            avg_data = [np.mean(time_unit) for time_unit in sim.T]

            # do statistics
            for k in dats:
                corr, _ = utils.StatsHandler.correlate(dats[k]['val'], avg_data)
                dats[k]['tmp'].append(corr)

            pbar.update(counter)
            counter += 1

        for k in dats: dats[k]['correlations'].append(dats[k]['tmp'])
    pbar.finish()

    for k in dats:
        present(
            '%s correlation development for increasing number of activating links (%s, %s)' % (k, Model.info['name'], 'time norm' if models.BooleanModel.info['norm_time'] else 'gene norm'), plotter.Plotter.errorbar_plot,
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


###############
# Experiments #
###############

def quot_investigator():
    investigate_active_edge_count_influence_quot(models.BooleanModel, scalefree=True, indegree=True)
    investigate_active_edge_count_influence_quot(models.BooleanModel, scalefree=True, indegree=None)
    investigate_active_edge_count_influence_quot(models.BooleanModel, scalefree=True, indegree=False)
    investigate_active_edge_count_influence_quot(models.BooleanModel, scalefree=False, indegree=True)
    investigate_active_edge_count_influence_quot(models.BooleanModel, scalefree=False, indegree=None)
    investigate_active_edge_count_influence_quot(models.BooleanModel, scalefree=False, indegree=False)

##################
# Command Center #
##################

if __name__ == '__main__':
    plotter.Plotter.show_plots = False
    logger.VERBOSE = True

    #simulate_model(models.MultiplicatorModel, runs=20)
    #simulate_model(models.BooleanModel)
    #simulate_model(models.LinearModel, plot_jc_ev=True, runs=20)
    #simulate_model(models.NonlinearModel, plot_jc_ev=True, runs=20)

    #analysis(utils.GraphGenerator.get_er_graph(100, 0.3), models.MultiplicatorModel)
    #gene_overview()

    #investigate_active_edge_count_influence_gene_expr(models.BooleanModel)
    #investigate_base_window_influence()
    #investigate_origin_of_replication_influence()
    #quot_investigator()

    #real_life_average()
    #real_life_all()
    #real_life_single()
    #real_life_rnaseq()
    rnaseq_vs_microarray()
