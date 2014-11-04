import numpy as np
import numpy.random as npr

from scipy.integrate import odeint

import networkx as nx

import parser


class Model(object):
    def __init__(self, graph):
        self.adja_m = graph.adja_m
        self.setup()

    def setup(self):
        pass

    def generate(self):
        pass

class BooleanModel(Model):
    """ Simple model to generate gene expression data
    """

    def setup(self):
        """ Declare activating and inhibiting links
        """
        # randomly inhibit or activate
        for y in range(self.adja_m.shape[0]):
            for x in range(self.adja_m.shape[1]):
                if self.adja_m[x,y] == 1:
                    self.adja_m[x,y] = npr.choice([-1, 1])

        # add self-inhibitory links for all nodes without incoming inhibitory links
        for x in range(self.adja_m.shape[1]):
            col = self.adja_m[:,x]
            col[col==1] = 0
            s = sum(abs(col))

            if s == 0:
                self.adja_m[x,x] = -1

    def generate(self, runs=10):
        """ Applies rule a couple of times and returns system evolution
        """
        def rule(x):
            """ Apply rule
            """
            return np.transpose(self.adja_m.dot(np.transpose(x)))

        num = self.adja_m.shape[0]
        x0 = npr.sample(num)

        data = np.matrix(x0)
        for t in range(runs):
            cur = rule(data[-1])
            data = np.vstack((data, cur))

        return np.array(data)

class ODEModel(Model):
    """ General model of ODEs to generate data
    """
    def setup(self):
        """ Setup needed constants
        """
        self.e1 = 0.2
        self.e2 = 0.2

    def generate(self, runs=10):
        """ Solves nonlinear system and returns solution
        """
        num = self.adja_m.shape[0]

        t = np.arange(0, runs, 1)
        x0 = npr.sample(num)

        def func(X, t=0):
            def iter(fun, i):
                """ Generates term using given nonlinear function
                """
                sigma = 0
                for j in range(num):
                    e = self.adja_m[i, j]
                    sigma += 0.5 * (abs(e) - e) * fun(X[j])
                return sigma
            terms = []

            for i in range(num):
                terms.append(self.e1 * iter(self.f1, i) + self.e1 * iter(self.f2, i))

            return np.array(terms)

        res = odeint(func, x0, t)
        return res

class LinearModel(ODEModel):
    def setup(self):
        super(NonlinearModel, self).setup()

        self.f1 = lambda x: x
        self.f2 = lambda x: -x

class NonlinearModel(ODEModel):
    def setup(self):
        super(NonlinearModel, self).setup()

        self.f1 = lambda x: 1/(1+x)
        self.f2 = lambda x: x/(1+x)


if __name__ == '__main__':
    import utils

    m = NonlinearModel(utils.GraphGenerator.get_random_graph())
    print(m.generate())
