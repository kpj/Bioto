import os.path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

import numpy as np

import networkx as nx

import utils, parser


g = utils.GraphHandler(
    utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')
)
#g.visualize('regnet.png')
#g.dump_adjacency_matrix('foo.txt')
#g.dump_node_names('names.txt')

perron_frobenius = None
while perron_frobenius == None:
    perron_frobenius = g.get_perron_frobenius()

fname = 'conc.bak'
if os.path.isfile('%s.npy' % fname):
    print('Recovering data from', fname)
    concentrations = np.load('%s.npy' % fname)
else:
    names = g.get_node_names()
    concentrations, fail = parser.parse_concentration(
        names,
        '../data/concentrations/GDS3597_full.soft'
    )
    concentrations = np.array(concentrations) / np.linalg.norm(concentrations)

    print('coverage', round(1 - len(fail)/len(names), 3))
    print('correlation', np.correlate(concentrations, perron_frobenius))

    # save for faster reuse
    np.save(fname, concentrations)

fig = plt.figure()
ax = plt.gca()

sc = sp = []
for c, p in zip(concentrations, perron_frobenius):
    if not (c == 0 or p == 0):
        sc.append(c)
        sp.append(p)

ax.loglog(sc, sp)

plt.title('Real-life comparisons')
plt.xlabel('Actual Concentrations')
plt.ylabel('Perron-Frobenius Eigenvector')

plt.show()
