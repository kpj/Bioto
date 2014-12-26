import os, os.path
import csv
import itertools

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

import networkx as nx

import plotter, utils


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


if __name__ == '__main__':
	plotter.Plotter.show_plots = True

	#plot_BM_runs('./BM_data')

	investigate_ER_edge_probs(100)