import numpy as np
import numpy.linalg as npl
import numpy.random as npr

from scipy.integrate import odeint


class Model(object):
    def __init__(self, graph):
        self.graph = graph
        self.setup()

        print('Applying model, this might change the adjacency matrix')

    def setup(self):
        pass

    def generate(self):
        Exception('This model lacks any generating function')

class MultiplicatorModel(Model):
    """ Simulates network evolution by adjacency matrix multiplication
    """

    def generate(self, runs=10):
        initial = np.array([npr.random() for i in range(len(self.graph))])

        data = [initial]
        for i in range(runs):
            cur = [self.graph.adja_m.dot(data[-1])[0, i] for i in range(len(self.graph))]
            cur /= npl.norm(cur)

            data = np.vstack((data, cur))

        return data

class BooleanModel(Model):
    """ Simple model to generate gene expression data
    """

    def setup(self):
        """ Declare activating and inhibiting links
        """
        # randomly inhibit or activate
        for y in range(self.graph.adja_m.shape[0]):
            for x in range(self.graph.adja_m.shape[1]):
                if self.graph.adja_m[x,y] == 1:
                    self.graph.adja_m[x,y] = npr.choice([-1, 1])

        # add self-inhibitory links for all nodes without incoming inhibitory links
        for x in range(self.graph.adja_m.shape[1]):
            col = self.graph.adja_m[:,x]
            col[col==1] = 0
            s = sum(abs(col))

            if s == 0:
                self.graph.adja_m[x,x] = -1

    def generate(self, runs=10):
        """ Applies rule a couple of times and returns system evolution
        """
        def rule(x):
            """ Apply rule
            """
            return np.transpose(self.graph.adja_m.dot(np.transpose(x)))

        num = self.graph.adja_m.shape[0]
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
        num = self.graph.adja_m.shape[0]

        t = np.arange(0, runs, 1)
        x0 = npr.sample(num)

        def func(X, t=0):
            def iter(fun, i):
                """ Generates term using given nonlinear function
                """
                sigma = 0
                for j in range(num):
                    e = self.graph.adja_m[i, j]
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
        super(LinearModel, self).setup()

        self.f1 = lambda x: x
        self.f2 = lambda x: -x

class NonlinearModel(ODEModel):
    def setup(self):
        super(NonlinearModel, self).setup()

        self.f1 = lambda x: 1/(1+x)
        self.f2 = lambda x: x/(1+x)