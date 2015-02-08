import os, os.path
import csv
import itertools

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from prettytable import PrettyTable

import networkx as nx

import pysoft

import plotter, utils, models


def plot_BM_runs(dire):
	""" Plots all runs of a BM simulation in order to find out if it's normally distributed
	"""
	runs = []

	# gather data
	for f in os.listdir(dire):
		runs.append([])

		fname = os.path.join(dire, f)
		with open(fname, 'r') as csvfile:
			reader = csv.reader(csvfile)

			for row in reader:
				runs[-1].append(row)

	# refactor data
	tmp = []
	for t in range(len(runs[0])):
		tmp.append([])
		for run in runs:
			tmp[-1].append(run[t])

	res = []
	for t in tmp:
		res.append(np.array(t).T)
	res = np.array(res)

	# plot data
	for t, time_point in enumerate(res):
		plots = []

		for g, entry in enumerate(time_point):
			"""plots.append({
				'x': range(len(entry)),
				'y': entry,
				'label': 'gene %d at time %d' % (g, t)
			})"""

			data = [float(s) for s in entry.tolist()]
			with open('BM_density_analysis', 'a') as fd:
				writer = csv.writer(fd)
				writer.writerow(data)

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
	graph = utils.GraphGenerator.get_random_graph(node_num=n, edge_prob=e)
	data = models.BooleanModel(graph).generate_binary_time_series().T
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
	col_names = ['name', 'origin', 'method', 'entry type', 'subset number', 'is timeseries?']
	table = PrettyTable(col_names)
	for n in col_names: table.align[n] = 'l'

	for f in os.listdir(dir):
		fname = os.path.join(dir, f)
		soft = pysoft.SOFTFile(fname)
		head = soft.header

		if len(head) == 0:
			continue

		sample_name = head['dataset']['name']
		origin = head['database']['database_name']
		data_type = head['dataset']['dataset_type']
		subsets = ', '.join([x['subset_description'] for x in head['dataset']['subsets']])
		is_time_series = 'min' in subsets

		table.add_row([sample_name, origin, data_type, subsets, len(head['dataset']['subsets']), is_time_series])

	print(table)
	if not table_fname is None:
		with open(table_fname, 'w') as fd:
			fd.write(str(table))


if __name__ == '__main__':
	plotter.Plotter.show_plots = True

	#plot_BM_runs('./BM_data')
	#investigate_ER_edge_probs(100)
	#visualize_discrete_bm_run(n=50)
	#simple_plot()
	list_data('../data/concentrations/', 'data_summary.txt')
