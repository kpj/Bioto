import operator

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


class Experiment(object):
    """ Basic class to deal with experiments.

        Each experiment is started by calling the conduct method.
        By convention, each autonomous step has to be encapsulated into a single function, prefixed with '_'.

        Each subclass must define "self.pathway" which lists all methods in the order they are supposed to be called.
    """
    def conduct(self):
        """ Call each experimental step in the right order
        """
        caller = self.__class__.__name__

        if caller == 'Experiment':
            raise NotImplementedError('This experiment has not been implemented yet')
        else:
            for func in self.pathway:
                print(func.__doc__)
                func()

class GeneExpressionVariance(Experiment):
    """ Observe how strong the signal in gene expression vectors compared to their shuffled counterpart is
    """
    def __init__(self, common_genes, data):
        self.common_genes = common_genes
        self.data = data

        self.x = []
        self.x_shuffled = []

        self.y = []
        self.y_shuffled = []

        self.variances = []
        self.variances_shuffled = []

        # this has to be done for any experiment in some form
        self.pathway = [self._generate_x, self._generate_y, self._compute_variances, self._create_plot]

    def _generate_x(self):
        """ Transform data such that x_i contains all gene expressions 1,...,j for experiment i
        """
        for exp in self.data:
            tmp = [t[1] for t in sorted(exp.items(), key=operator.itemgetter(0))]
            self.x.append(tmp)

		# create shuffled version to check importance of signal in actual data
        for e in self.x:
            self.x_shuffled.append(npr.permutation(e))

    def _generate_y(self):
        """ Generate y(m), which contains y_j(m) for j=1,...,n, which in turn contains the gene expression levels obtained by averaging the first m experiments
        """
        def gen_y(m, gene_data):
            foo = [] # average gene expression over experiments
            for g in range(len(self.common_genes)):
                # tmp contains average gene expression (for one gene) over m experiments
                tmp = 1/m * sum([float(gene_data[i][g]) for i in range(m)])
                foo.append(tmp)
            return foo

        for m in range(1, len(self.data)):
            self.y.append(gen_y(m + 1, self.x))
            self.y_shuffled.append(gen_y(m + 1, self.x_shuffled))

    def _compute_variances(self):
        """ Compute variances of shuffled and ordered gene expression data
        """
        for e, e_shuffled in zip(self.y, self.y_shuffled):
            self.variances.append(np.var(e))
            self.variances_shuffled.append(np.var(e_shuffled))

    def _create_plot(self):
        """ Create plot of generated data
        """
        t = range(len(self.variances))

        plt.plot(t, self.variances, label='actual data')
        plt.plot(t, self.variances_shuffled, label='shuffled gene expression vector')

        plt.title('Variance overview of gene expression data')
        plt.xlabel('number of considered data sets')
        plt.ylabel('variance')
        plt.legend(loc='best')

        plt.show()
