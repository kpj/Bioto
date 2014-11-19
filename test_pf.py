""" Check perron-frobenuis eigenvector computations
    Run with: `while true ; do python test_pf.py && break ; done`
    Don't forget to delete 'stuff.nby' for a restart.
"""

import sys

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt
import numpy.random as npr

import matplotlib.pyplot as plt

import utils


# config
runs = 20

# graph
g = utils.GraphGenerator.get_regulatory_graph('../data/architecture/network_tf_gene.txt')

# handle caching
try:
    mat = np.load('stuff.npy').tolist()
except FileNotFoundError:
    mat = []

for i in range(runs - len(mat)):
    # pf
    val, vec = npl.eig(g.adja_m) # returns already normalized eigenvectors
    max_eigenvalue_index = np.argmax(np.real(val))
    perron_frobenius = np.array(np.transpose(np.real(vec[:, max_eigenvalue_index])).tolist()[0])

    if all(i <= 0 for i in perron_frobenius):
        print("Rescaled pf-eigenvector by -1")
        perron_frobenius *= -1
    elif any(i < 0 for i in perron_frobenius):
        print("Error, pf-eigenvector is malformed")

    perron_frobenius /= npl.norm(perron_frobenius, 1)

    # error
    diff = []
    av = g.adja_m.dot(perron_frobenius).tolist()[0]
    lv = val[max_eigenvalue_index] * perron_frobenius
    for i, j in zip(av, lv):
        diff.append(abs(i - np.real(j)))

    # save it for later
    mat.append(diff)
    np.save('stuff', mat)

    break
else:
    # plot
    ax = plt.gca()

    for diff in mat:
        ax = plt.gca()

        ax.loglog(
            range(len(diff)), diff,
            linestyle='None',
            marker='.', markeredgecolor='blue'
        )

    ax.set_title('Errors in pf-computation of e.coli. network')
    ax.set_xlabel('Index of pf-component')
    ax.set_ylabel('Error = abs(A*v - l*v), v:eivec, l:eival')

    plt.show()
    sys.exit(0)
sys.exit(42)
