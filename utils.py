import re
import copy
import collections
import os, os.path
import operator
import json, csv
import hashlib
import random
import datetime

import numpy as np
import numpy.random as npr

import scipy.stats as scits
import pandas as pd
import networkx as nx

from progressbar import ProgressBar

import pysoft

import file_parser, graph, errors
from logger import log


class GraphGenerator(object):
    """ Return different types of network configurations (i.e. topologies, etc)
    """
    @staticmethod
    def get_er_graph(node_num=20, edge_prob=0.3):
        """ Return simple ER-graph
        """
        return graph.Graph(nx.erdos_renyi_graph(node_num, edge_prob, directed=True))

    @staticmethod
    def get_scalefree_graph(node_num=20):
        return graph.Graph(nx.scale_free_graph(node_num))

    @staticmethod
    def get_random_graph(graphdef, activating_edges=0, inhibiting_edges=0, scalefree=False):
        """ Create random graph by subsetting complete graph
            If graphdef is an integer, this will create an ER graph
        """
        if isinstance(graphdef, int):
            node_num = graphdef

            if scalefree:
                while True:
                    # brute-force right amount of edges, this might fail horribly!
                    g = GraphGenerator.get_scalefree_graph(graphdef).graph
                    if len(g.edges()) >= activating_edges+inhibiting_edges: break
            else:
                g = nx.complete_graph(graphdef, create_using=nx.DiGraph())
        elif isinstance(graphdef, graph.Graph):
            node_num = len(graphdef)
            g = graphdef.graph
        else:
            raise RuntimeError('Invalid graph definition given')

        edges = random.sample(g.edges(), activating_edges + inhibiting_edges)
        a_edges = edges[:activating_edges]
        i_edges = edges[-inhibiting_edges:] if inhibiting_edges > 0 else []

        amat = np.zeros((node_num, node_num))
        for source, sink in a_edges: amat[source, sink] = 1
        for source, sink in i_edges: amat[source, sink] = -1

        gg = graph.Graph(nx.from_numpy_matrix(abs(amat), create_using=nx.DiGraph()))
        gg.aug_adja_m = amat

        return gg

    @staticmethod
    def get_regulatory_graph(tf_reg_file, gene_proximity_file=None, base_window=50000, reduce_gpn=True, origin=None):
        """ Return transcriptional regulatory network specified by given file
            If gene_proximity_file is given, TRN and GPN will be composed to a MultiDiGraph
        """
        log('Generating TRN')
        trn = graph.Graph(file_parser.generate_tf_gene_regulation(tf_reg_file), largest=True)
        log(' > %d nodes' % len(trn.graph.nodes()))
        log(' > %d edges' % len(trn.graph.edges()))

        if not gene_proximity_file is None:
            log('Generating GPN', end=' ', flush=True)
            gpn = GraphGenerator.get_gene_proximity_network(gene_proximity_file, base_window, origin=origin)
            log(' > %d nodes' % len(gpn.graph.nodes()))
            log(' > %d edges' % len(gpn.graph.edges()))
            if reduce_gpn: gpn.reduce_to(trn)

            trn += gpn

        return trn

    @staticmethod
    def get_gene_proximity_network(gene_prox_file, base_window, origin=None):
        """ Return gene proximity network specified by given file
        """
        gpng = file_parser.GPNGenerator(gene_prox_file)
        if origin is None:
            log('(circular genome)')
            gpn = gpng.generate_gene_proximity_network_circular(base_window)
        else:
            log('(two-stranded genome)')
            gpn = gpng.generate_gene_proximity_network_two_strands(base_window, origin)

        return graph.Graph(gpn)

class StatsHandler(object):
    FUNC = scits.pearsonr
    #FUNC = scits.spearmanr

    @staticmethod
    def correlate(x, y, compute_bands=False):
        """ Computes Pearson coefficient of x, y and compares it to the correlation of shuffled forms of x, y
        """
        (corr, p_val) = StatsHandler.FUNC(x, y) # correlation r [-1,1], prob. that null-hypothesis (i.e. x,y uncorrelated) holds [0,1]

        if not compute_bands:
            return corr, p_val

        xs = np.copy(x)
        ys = np.copy(y)

        rs = []
        for i in range(3333):
            npr.shuffle(xs)
            npr.shuffle(ys)

            (r, p) = StatsHandler.FUNC(xs, ys)
            rs.append(r)

        mi, ma = min(rs), max(rs)
        return corr, p_val, mi, ma

class DataHandler(object):
    backup_dir = 'conc_baks'

    @staticmethod
    def _handle_data(graph, exp):
        """ Take gene expression dict and return gene expression vector and used indices
        """
        used_genes = set()
        tmp = collections.defaultdict(dict)
        for gene in graph:
            for col in exp.get_columns():
                if gene in exp.get_genes_in_col(col):
                    tmp[col][gene] = exp.data[col][gene]
                    used_genes.add(gene)
        exp.data = dict(tmp)

        #matched = set(graph).intersection(used_genes)
        no_match = set(graph).difference(used_genes)
        log(' > graph coverage:', round(1 - len(no_match)/len(graph), 3))

        exp.clear_genes()
        exp.add_genes(used_genes)

        return exp

    @staticmethod
    def load_rnaseq_data(graph, fname, **kwargs):
        log('Parsing RNAseq data file "%s"' % fname)
        data = file_parser.parse_rnaseq(fname, **kwargs)
        return DataHandler._handle_data(graph, data)

    @staticmethod
    def load_concentrations(graph, fname, conc_range=None, **kwargs):
        """ Extract gene concentrations from file which also appear in graph.
            Also return vector of node indices used in concentration vector
        """
        bak_fname = os.path.join(DataHandler.backup_dir, 'conc_%s.bak' % os.path.basename(fname))

        if os.path.isfile('%s.npy' % bak_fname):
            log('Recovering data from "%s"' % bak_fname)
            exp = np.load('%s.npy' % bak_fname).item()
        else:
            log('Parsing SOFT data file "%s"' % fname)
            gdsh = GDSHandler(os.path.dirname(fname))
            data = gdsh.parse_file(os.path.basename(fname), conc_range, **kwargs)

            exp = DataHandler._handle_data(graph, data)

            # cache for faster reuse
            if not os.path.exists(DataHandler.backup_dir):
                os.makedirs(DataHandler.backup_dir)
            np.save(bak_fname, exp)

        return exp

    @staticmethod
    def load_averaged_concentrations(graph, directory, conc_range=None, cache_file=None):
        """ Load concentration files in given directory and average them.
            In order to account for genes which appear in the graph but not in any dataset, this function will also return a vector of indices of the genes of the graph which were found in at least on dataset
        """
        # gather all data
        gdsh = GDSHandler(directory)
        experiments = gdsh.process_directory(conc_range=conc_range)

        # accumulate data
        res = GDSParseResult(['average'])

        # remove previous data file
        if not cache_file is None:
            if os.path.isfile(cache_file):
                os.remove(cache_file)

        log('Averaging data')
        pbar = ProgressBar(maxval=len(graph))
        pbar.start()

        used_genes = set()
        dataset_lens = []
        for i, gene in enumerate(graph):
            gene_concs = []
            for exp in experiments:
                for col in exp.get_columns():
                    if gene in exp.get_genes_in_col(col):
                        gene_concs.append(exp.data[col][gene])
                        used_genes.add(gene)

                        res.add_filename(exp.filename)

            if not cache_file is None:
                with open(cache_file, 'a') as fd:
                    writer = csv.writer(fd)
                    writer.writerow([gene] + gene_concs)

            pbar.update(i)
            if len(gene_concs) == 0: continue

            dataset_lens.append(len(gene_concs))
            res.data['average'][gene] = np.mean(gene_concs)
        pbar.finish()

        res.add_genes(used_genes)
        log(' > entries/gene (avg): %f' % np.mean(dataset_lens))

        exp = DataHandler._handle_data(graph, res)
        return exp

class CacheHandler(object):
    cache_directory = 'plot_data'

    @staticmethod
    def store_plot_data(title, func, *args, model=None, **kwargs):
        """ Store plot data and return sanitized data dict
        """
        if len(args) == 2:
            """ args = (x_label, data)
            """
            dic = {
                'data': args[1],
                'title': title,
                'x_label': args[0],
            }
        elif len(args) == 3:
            """ args = (x_label, y_label, data)
            """
            dic = {
                'data': args[2],
                'title': title,
                'x_label': args[0],
                'y_label': args[1]
            }
        elif len(args) == 4:
            """ args = (x_label, x_data, y_label, y_data)
                args = (x_label, x_data, y_label, [(label, y_data), ..])
            """
            dic = {
                'x_data': args[1],
                'y_data': args[3],
                'title': title,
                'x_label': args[0],
                'y_label': args[2]
            }

        dic['info'] = {}
        dic['info']['function'] = func.__name__

        model_id = 'raw'
        if not model is None:
            dic['info'].update(model.info)
            model_id = model.hash()

        if 'args' in dic:
            dic['args'].update(kwargs)
        else:
            dic['args'] = kwargs

        if not os.path.exists(CacheHandler.cache_directory):
            os.makedirs(CacheHandler.cache_directory)
        CacheHandler.dump('%s__%s.dat' % (os.path.join(CacheHandler.cache_directory, clean_string(title)), model_id), dic)

        return dic

    @staticmethod
    def dump(fname, obj):
        """ Save dictionary to file (convert numpy arrays to python lists)
        """
        def clean(foo):
            if isinstance(foo, dict):
                return {k: clean(v) for (k,v) in foo.items()}
            elif isinstance(foo, list):
                return [clean(e) for e in foo]
            elif type(foo) == type(np.array([])): # TODO: fixme
                return clean(foo.tolist())
            elif isinstance(foo, range):
                return list(foo)
            return foo

        with open(fname, 'w') as fd:
            json.dump(clean(obj), fd)

    @staticmethod
    def load(fname):
        with open(fname, 'r') as fd:
            return json.load(fd)

class GDSHandler(object):
    """ Handles directories containing many GDS files
    """
    def __init__(self, dirname):
        self.dir = dirname

        self.all_genes = None # genes which appear in at least one dataset
        self.common_genes = None # genes which appear in all datasets

    def parse_file(self, fname, conc_range=None, **kwargs):
        """ Extract all (valid) gene concentrations from file
            fname is relative to dirname given in constructor
        """
        return file_parser.parse_concentration(os.path.join(self.dir, fname), conc_range, **kwargs)

    def process_directory(self, only_common_genes=False, conc_range=None):
        """ Scan directory for SOFT files.
            Extract gene concentration of genes which appear in all datasets if requested
        """
        log('Parsing SOFT files in "%s"' % self.dir)
        walker = list(os.walk(self.dir))
        pbar = ProgressBar(maxval=sum([len(files) for root, dirs, files in walker]))
        pbar.start()

        self.all_genes = set()
        self.quasi_genes = set()
        self.common_genes = None

        experiments = []
        counter = 0
        for root, dirs, files in walker:
            for fname in sorted(files):
                counter += 1
                pbar.update(counter)
                if not is_soft_file(fname): continue

                try:
                    res = self.parse_file(fname, conc_range=conc_range)
                except errors.InvalidGDSFormatError:
                    continue

                experiments.append(res)

                self.all_genes.update(set(res.get_genes()))
                self.quasi_genes.update(set(res.get_common_genes()))

                cgenes = res.get_common_genes()
                if len(cgenes) != 0:
                    if self.common_genes is None:
                        self.common_genes = cgenes
                    else:
                        self.common_genes = set(cgenes).intersection(self.common_genes)
        pbar.finish()

        self.all_genes = sorted(self.all_genes) # genes that appear in any column of any file
        self.quasi_genes = sorted(self.quasi_genes) # genes that appear in all columns of any files
        self.common_genes = sorted(self.common_genes) # genes that appear in all columns of all files

        # extract gene concentrations
        genes_to_extract = self.common_genes if only_common_genes else self.all_genes
        for exp in experiments:
            for col in exp.data:
                genes = exp.data[col].keys()

                tmp = {k: exp.data[col][k] if k in genes else None for k in genes_to_extract}
                tmp = dict(filter(lambda x: not x[1] is None, tmp.items()))

                exp.data[col] = tmp

        return experiments

class GDSFormatHandler(object):
    """ Handle all the different file formats GDS files may come in
    """
    def __init__(self, soft, throw_on_unknown_format=True):
        self.soft = soft
        self.type = self.soft.header['dataset']['dataset_value_type']
        self.throw_on_unknown_format = throw_on_unknown_format

        self.column_keywords = ['aerobic', 'anaerobic']

    def get_useful_columns(self):
        """ Return list of columns which contain data which can be used (i.e. not mutants, but rather wild types under different external conditions)
        """
        # get sample descriptions
        data = {}
        for subset in self.soft.header['dataset']['subsets']:
            for ss_id in subset['subset_sample_id'].split(','):
                data[ss_id] = subset['subset_description'].lower()

        # extract columns according to fitting sample descriptions
        cols = []
        for name, desc in data.items():
            if 'mutant' in desc: continue

            if (
                'wild type' in desc or # default
                'control' in desc or # default
                desc in self.column_keywords or # specific set of keywords
                re.match(r'[0-9]+ min(utes)?', desc) or # time series
                re.match(r'ph [0-9]+', desc) or # pH value
                re.match(r'od(600)? [-+]?[0-9]*\.?[0-9]*', desc), # some optical density stuff
                re.match(r'[0-9]+ ug/ml', desc)
            ):
                cols.append(name)

        return cols

    def get_data(self):
        """ Yield data in soft file after transforming as needed
        """
        for row in self.soft.data:
            yield self.parse_row(row)

    def transform_row(self, r, func):
        """ Transform all elements in new instance of given row by given operator if possible
        """
        row = copy.copy(r)

        for i in range(len(row)):
            try:
                row[i] = func(row[i])
            except:
                pass

        return row

    def parse_row(self, row):
        """ Transform all data to common format
        """
        if self.type == 'log2 ratio':
            return row

        if self.throw_on_unknown_format:
            raise errors.InvalidGDSFormatError('Encountered invalid format "%s"' % self.type)
        else:
            return row

class GDSParseResult(object):
    def __init__(self, conc_range=[]):
        self.data = {c: {} for c in conc_range}

        self._genes = set()
        self.filename = None

    def __eq__(self, other):
        return self.data == other.data and self._genes == other._genes and self.filename == other.filename

    def trim_input(self, inp, graph, col):
        """ Extract elements from input whose position corresponds to genes used in this dataset and actually appearing in the graph
        """
        used_gene_indices = [list(graph).index(gene) for gene in self.get_genes_in_col(col)]
        return [inp[i] for i in used_gene_indices]

    def get_data(self, cols=None):
        """ Return column and contained data
        """
        for col in sorted(self.data if cols is None else cols):
            conc = [t[1] for t in sorted(self.data[col].items(), key=operator.itemgetter(0))]

            yield col, conc

    def get_columns(self):
        for col in sorted(self.data.keys()):
            yield col

    def get_genes_in_col(self, col):
        for gene in sorted(self.data[col].keys()):
            yield gene

    def add_gene(self, gene):
        self._genes.add(gene)

    def add_genes(self, genes):
        for g in genes:
            self.add_gene(g)

    def get_genes(self):
        """ Return genes present in any column
        """
        return sorted(self._genes)

    def get_common_genes(self):
        """ Return genes present in all columns
        """
        genes = []
        for col in sorted(self.data):
            res = self.data[col].keys()
            if len(res) != 0: genes.append(set(res))

        return sorted(set.intersection(*genes))

    def clear_genes(self):
        self._genes = set()

    def add_filename(self, fname):
        if self.filename is None:
            self.filename = fname
        else:
            if not fname in self.filename:
                foo = self.filename.split(',')
                foo.append(fname)
                self.filename = ','.join(foo)


def combine_gds_parse_results(res1, res2, col):
    """ Return data in given column for genes which appear in both results
    """
    if isinstance(col, tuple):
        common_genes = list(sorted(set.intersection(*[set(vec.data[c].keys()) for c, vec in zip(col, [res1, res2])])))

        cols = col
    else:
        common_genes = list(sorted(set.intersection(*[set(vec.data[col].keys()) for vec in [res1, res2]])))

        cols = (col, col)

    vec1 = [res1.data[cols[0]][g] for g in common_genes]
    vec2 = [res2.data[cols[1]][g] for g in common_genes]

    return (vec1, vec2)

def clean_string(s):
    """ Make string useable as filename
    """
    return s.replace(' ', '_')

def df(**kwargs):
    """ Convert given args to proper dataframe
    """
    dic = {k: pd.Series(v, dtype='category') for (k,v) in kwargs.items()} # TODO: handle category properly
    return pd.DataFrame(dic)

def md5(s):
    m = hashlib.md5()
    m.update(s.encode(encoding='utf-8'))
    return m.hexdigest()

def is_soft_file(fname):
    return fname.endswith('.soft') or fname.endswith('.soft.gz')

def get_interquartile_variance(data, pop_range=[25, 75]):
    irange = np.percentile(data, pop_range, interpolation='nearest')

    lspace = np.linspace(irange[0], irange[1], 2)
    l_res = np.digitize(data, lspace)
    r_res = np.digitize(data, lspace, right=True)

    l_idata = set([data[i] for i in range(len(data)) if l_res[i] == 1])
    r_idata = set([data[i] for i in range(len(data)) if r_res[i] == 1])

    return np.var(list(l_idata.union(r_idata)))

def get_max_entry_index(vec, real_entries_only=False):
    """ Return index of element with highest real part (neglect complex numbers if wanted)
    """
    max_index = None
    if real_entries_only:
        lv = -float('inf')
        for i, v in enumerate(vec):
            if np.isreal(v):
                if v > lv:
                    max_index = i
                    lv = v
    else:
        max_index = np.argmax(np.real(vec))

    return max_index

def get_all_data(g):
    """ Return all concentrations, PFs and PRs
    """
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

    return conc_vec, pf_vec, pr_vec

def get_strtime():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')
