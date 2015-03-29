import os, os.path, shutil
import csv
import itertools
import json
import collections
import sys
import operator

import numpy as np
import numpy.random as npr
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from prettytable import PrettyTable
from progressbar import ProgressBar

import networkx as nx

import pysoft

import plotter, utils, models, graph, file_parser, experiment_classes


def investigate_ER_edge_probs(node_num):
	""" Compare theoretical versus practical edge numbers in ER-graph
	"""
	def max_edge_num(nodes):
		return (nodes*(nodes-1)/2)

	# check property for some ER-graphs
	en = max_edge_num(node_num)
	for p in np.arange(0, 1, 0.2):
		graph = nx.erdos_renyi_graph(node_num, p)
		edge_num = len(graph.edges())
		pp = edge_num / en

		print(p, '->', pp)
	print()

	# get probability for TRN
	trn = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')

	trn_node_num = len(trn.graph.nodes())
	trn_edge_num = len(trn.graph.edges())
	p = trn_edge_num / max_edge_num(trn_node_num)

	print(p)

def visualize_discrete_bm_run(n=100, e=0.3):
	""" Visualize one discrete BM run
	"""
	graph = utils.GraphGenerator.get_er_graph(node_num=n, edge_prob=e)

	model = models.BooleanModel(graph)
	#model.aug_adja_m = np.array([[-1,1,0,1,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0,-1,0],[-1,0,0,0,0,0,-1,0,0,0],[0,0,0,0,-1,0,0,0,0,1],[-1,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,-1,0,0,0],[0,0,0,0,0,0,0,-1,0,0],[1,0,0,-1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,-1]])
	data = model.generate_binary_time_series().T #initial_state=[0,1,0,0,1,0,0,1,1,0]

	plotter.Plotter.plot_heatmap(data, 'Characteristic evolution of discrete Boolean model', 'time', 'node')

def simple_plot():
	""" Plot 1/(1+x) and x/(1+x) [functions used in nonlinear model]
	"""
	x = np.arange(0, 5.1, 0.1)
	plt.tick_params(labelsize=20)

	f1 = lambda x: 1/(1+x)
	f2 = lambda x: x/(1+x)

	plt.plot(x, f1(x), label=r'$\frac{1}{1+x}$')
	plt.plot(x, f2(x), label=r'$\frac{x}{1+x}$')

	plt.title(r'Plot of $f^*_i$ used in Nonlinear model', fontsize=33)
	plt.xlabel('x', fontsize=30)
	plt.ylabel('y', fontsize=30)

	plt.legend(loc='best')

	fig = plt.gcf()
	fig.savefig('f.png', dpi=150)

def list_data(dir, table_fname=None):
	""" Generate overview over all SOFT files present in given directory
	"""
	col_names = ['name', 'origin', 'method', 'entry type', 'value type', 'is timeseries?']
	table = PrettyTable(col_names)
	for n in col_names: table.align[n] = 'l'

	for f in sorted(os.listdir(dir)):
		fname = os.path.join(dir, f)
		soft = pysoft.SOFTFile(fname, skip_data=True)
		head = soft.header

		if len(head) == 0:
			continue

		sample_name = head['dataset']['name']
		origin = head['database']['database_name']
		data_type = head['dataset']['dataset_type']
		subsets = '[%d]: %s' % (len(head['dataset']['subsets']), ', '.join([x['subset_description'] for x in head['dataset']['subsets']]))
		is_time_series = 'min' in subsets
		dataval_type = head['dataset']['dataset_value_type']

		table.add_row([sample_name, origin, data_type, subsets, dataval_type, is_time_series])

	print(table)
	if not table_fname is None:
		with open(table_fname, 'w') as fd:
			fd.write(table.get_string())

def search_database(dbname, save_dir=None, stats_file=None):
	""" Crawl given database for SOFT files which comply with the given criterion
	"""
	walker = list(os.walk(dbname))
	pbar = ProgressBar(maxval=len(walker))
	stats_dict = collections.defaultdict(list)
	used_gds = set()

	if not save_dir is None:
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

	pbar.start()
	for i, (root, dirs, files) in enumerate(walker):
		for fname in files:
			if '.soft' in fname:
				fn = os.path.join(root, fname)
				soft = pysoft.SOFTFile(fn, skip_data=True)
				org = soft.header['dataset']['dataset_platform_organism'].lower()

				# don't handle duplicate entries (e.g. '_full')
				gds = soft.header['dataset']['name']
				if gds in used_gds: continue
				used_gds.add(gds)

				stats_dict[org].append(fname)

				if save_dir is None: continue
				if 'coli' in org:
					try:
						shutil.copy(fn, save_dir)
					except shutil.SameFileError:
						print('Trying to overwrite file, please restart with clean state')
						sys.exit(1)

		pbar.update(i)
	pbar.finish()

	if not stats_file is None:
		json.dump(stats_dict, open(stats_file, 'w'))

def plot_orga_distri(distr_file, fout):
	""" Parse stats file created by database crawler to get absolute counts in csv format
	"""
	dat = json.load(open(distr_file, 'r'))
	csv_file = 'geo_org_distr.csv'

	with open(csv_file, 'w') as fd:
		for organism, gds_files in sorted(dat.items()):
			fd.write('%s,%s\n' % (organism, len(gds_files)))

	os.system('Rscript org_distr_plot.R "%s"' % fout)

def variance_of_gene_expression(data_dir):
	""" Analyze the composition of real-life GEO data
	"""
	save_file = 'gene_variance.dat'
	if not os.path.isfile('%s.npy' % save_file):
		gdsh = utils.GDSHandler(data_dir)

		experis = gdsh.process_directory(only_common_genes=True)
		common_genes = gdsh.common_genes

		np.save(save_file, (common_genes, experis))
	else:
		print('Using cached data')
		common_genes, experis = np.load('%s.npy' % save_file)

	experiment = experiment_classes.GeneExpressionVariance(common_genes, experis)
	experiment.conduct(shuffle_experiment_order=True)

def summarize_gene_expression_data(dirname):
	""" Show histogram for all SOFT files (also those with unsupported formats)
	"""
	gdsh = utils.GDSHandler(dirname)

	res = []
	for root, dirs, files in os.walk(dirname):
		for fname in sorted(files):
			if not utils.is_soft_file(fname): continue

			pres = gdsh.parse_file(fname, throw_on_unknown_format=False)

			for col, conc in pres.get_data():
				if len(conc) == 0: continue
				res.append({
					'data': conc,
					'fname': '%s:%s' % (pres.filename, col)
				})

	for e in res:
		plotter.Plotter.plot_histogram({'data': e['data'], 'title': 'Gene expression vector overview for %s' % e['fname'], 'x_label': 'gene expression value', 'y_label': 'count'})


if __name__ == '__main__':
	plotter.Plotter.show_plots = True

	#investigate_ER_edge_probs(100)
	#visualize_discrete_bm_run(n=10)
	#simple_plot()
	#list_data('../data/concentrations/', 'data_summary.txt')
	#search_database('/home/kpj/GEO/ftp.ncbi.nlm.nih.gov', save_dir='/home/kpj/GEO/ecoli', stats_file='GDS_stats.json')
	#plot_orga_distri('GDS_stats.json', 'geo_db_organism_distribution.png')
	#variance_of_gene_expression('/home/kpj/GEO/ecoli')
	summarize_gene_expression_data('../data/concentrations/')
