import os, os.path
import csv

import numpy as np
import numpy.linalg as npl
import numpy.random as npr

from scipy.integrate import odeint

import sympy as sp
from sympy.utilities.lambdify import lambdify

import utils, stats


class Math(object):
    def __init__(self, model):
        self.model = model

    def get_augmented_adja_m(self, edge_type=None):
        """ Return augmented adjacency matrix, i.e.
            1: activating, -1: inhibiting
        """
        # only create one unique augmented adjacency matrix
        if not self.model.aug_adja_m is None:
            print('Augmented adjacency matrix already computed, using old one.')
            return self.model.aug_adja_m

        # only create new one, if graph has not been assigned a specific one
        if not self.model.graph.aug_adja_m is None:
            print('Graph has been assigned an augmented adjacency matrix. Model uses that one.')
            return self.model.graph.aug_adja_m

        aug_adja_m = np.copy(self.model.graph.adja_m)

        # randomly inhibit or activate
        for y in range(len(self.model.graph)):
            for x in range(len(self.model.graph)):
                if aug_adja_m[x,y] == 1:
                    aug_adja_m[x,y] = npr.choice([-1, 1]) if edge_type is None else edge_type

        # add self-inhibitory links for all nodes without incoming inhibitory links
        for x in range(len(self.model.graph)):
            col = np.copy(aug_adja_m[:,x])
            col[col==1] = 0
            s = sum(abs(col))

            if s == 0:
                aug_adja_m[x,x] = -1

        return aug_adja_m.T

    def get_jacobian_ev(self, point):
        """ Return eigenvector of highest eigenvalue of jacobian
        """
        syms = [sp.Symbol('x'+str(i)) for i in range(len(self.model.graph))]
        terms = self.model.get_terms(syms)

        # generate/evaluate jacobian
        mat = []
        for t in terms:
            mat.append([])
            for s in syms:
                f = lambdify(syms, t.diff(s), 'numpy')
                mat[-1].append(f(*point))
        mat = np.matrix(mat)

        # do eigenanalysis
        ev = self.model.graph.math.get_perron_frobenius(mat=mat, test_significance=False, real_eva_only=True, rescale=False)
        return ev

class Model(object):
    info = {
        'name': 'vanilla'
    }

    @classmethod
    def hash(cls):
        """ Give unique, consistent hash of this model configuration
        """
        return utils.md5(repr(sorted(cls.info.items())))

    def __init__(self, graph):
        self.graph = graph
        self.math = Math(self)

        self.setup()

    def setup(self):
        self.aug_adja_m = None
        self.stats = None

    def generate(self, **kwargs):
        Exception('This model lacks any generating function')

class MultiplicatorModel(Model):
    """ Simulates network evolution by adjacency matrix multiplication
    """
    info = {
        'name': 'Multiplicator Model'
    }

    def generate(self, **kwargs):
        initial = np.array([npr.random() for i in range(len(self.graph))])

        data = [initial]
        for i in range(kwargs['runs']-1):
            cur = [self.graph.adja_m.dot(data[-1])[0, i] for i in range(len(self.graph))]
            cur /= npl.norm(cur)

            data = np.vstack((data, cur))

        return data

class BooleanModel(Model):
    """ Simple model to generate gene expression data
    """
    data_dir = 'BM_data'
    info = {
        'name': 'Boolean Model',
        'norm_time': True, # norm along time or gene axis
        'max_bin_mod_runs': 15, # how often to run the binary simulation (ON/OFF genes)
        'time_window': 30, # time window to average binary runs
        'cont_evo_runs': 300, # how many binary runs to generate
        'avg_runs': 1 # how many averaged binary runs to average in the end
    }

    def setup(self):
        """ Declare activating and inhibiting links and further constants
        """
        Model.setup(self)
        self.aug_adja_m = self.math.get_augmented_adja_m()
        self.stats = stats.BMStatsMan()

    def generate_binary_time_series(self, initial_state=None):
        """ Applies rule a couple of times and returns system evolution as binary ON/OFF states
            Returns:
            [
                [g1, g2, g3, ...], # at t0
                [g1, g2, g3, ...], # at t1
                ...
            ]
        """
        def rule(x):
            """ Apply rule
            """
            xnext = []
            x = x.tolist()[0]

            for i in range(len(x)):
                sigma = 0
                for j in range(len(x)):
                    sigma += self.aug_adja_m[i,j] * x[j]

                res = -1
                if sigma > 0:
                    res = 1
                elif sigma < 0:
                    res = 0
                else:
                    res = x[i]

                xnext.append(res)

            return xnext

        num = self.aug_adja_m.shape[0]
        x0 = npr.randint(2, size=num) if initial_state is None else initial_state

        data = np.matrix(x0)
        for t in range(BooleanModel.info['max_bin_mod_runs']):
            cur = rule(data[-1])
            data = np.vstack((data, cur))

            #if (cur == data[-2]).all():
            #    self.stats.early_stops += 1
            #    break

        return np.array(data)

    def generate_continues_evolution(self, norm_time):
        """ Generates continuous data from binary evolution
            Either norms along gene- or time-axis
            Returns:
            [
                [g1, g2, g3, ...], # at t0
                [g1, g2, g3, ...], # at t1
                ...
            ] # also format of "data", "concs"
        """
        data = np.array([])

        # concatenate time series
        for i in range(BooleanModel.info['cont_evo_runs']):
            self.stats.discrete_runs += 1

            res = self.generate_binary_time_series()
            data = np.vstack((data, res)) if len(data) > 0 else res

        # average gene activations over time window
        concs = []
        for i in range(0, len(data), BooleanModel.info['time_window']):
            concs.append([])
            for n in range(len(self.graph)):
                window = data.T[n,i:i+BooleanModel.info['time_window']]
                concs[-1].append(window.mean())
        concs = np.array(concs)

        # normalize along genes/time
        out = []
        if norm_time:
            for time in concs.T:
                norm = npl.norm(time)
                time /= norm
                out.append(time)
            out = np.array(out).T
        else:
            for genes in concs:
                norm = npl.norm(genes)
                genes /= norm
                out.append(genes)
            out = np.array(out)

        return out

    def generate(self, **kwargs):
        """ Averages many runs and returns result

            Parameters:
                **kwargs - Possible keys: 'cache'
            Returns:
                Averaged Boolean model data of the form
                [
                    [g1, g2, g3, ...], # at t0
                    [g1, g2, g3, ...], # at t1
                    ...
                ]
        """
        if 'cache' in kwargs and kwargs['cache']:
            if not os.path.exists(BooleanModel.data_dir):
                os.makedirs(BooleanModel.data_dir)

        tmp = None
        for i in range(BooleanModel.info['avg_runs']):
            cur = self.generate_continues_evolution(norm_time=BooleanModel.info['norm_time'])
            if not tmp is None:
                tmp += cur
            else:
                tmp = cur

            if 'cache' in kwargs and kwargs['cache']:
                with open('BM_data/BM_data_%02d.csv' % i, 'a') as fd:
                    writer = csv.writer(fd)
                    writer.writerows(cur)

        #self.stats.info()
        return tmp / BooleanModel.info['avg_runs']

class ODEModel(Model):
    """ General model of ODEs to generate data
    """
    def setup(self):
        """ Setup needed constants,
            this method needs to be called by all children
        """
        Model.setup(self)

        self.e1 = 0.1
        self.e2 = 0.9

        # some defaults, should be overwritten later on
        self.f1 = lambda x: 1
        self.f2 = lambda x: 1

        self.aug_adja_m = self.math.get_augmented_adja_m()

    def get_terms(self, X):
        """ Return individual terms of ODE.
            This function may be overwritten in order to change the systems behaviour
        """
        def iter(fun, i, fac):
            """ Generates term using given specified functions
            """
            sigma = 0
            for j in range(len(self.graph)):
                e = self.aug_adja_m[i, j]
                sigma += 0.5 * (abs(e) + fac * e) * fun(X[j])
            return sigma
        terms = []

        for i in range(len(self.graph)):
            terms.append(self.e1 * iter(self.f1, i, -1) + self.e2 * iter(self.f2, i, 1))

        return np.array(terms)

    def generate(self, **kwargs):
        """ Solves nonlinear system and returns solution
        """
        t = np.arange(0, kwargs['runs'], 1)
        x0 = npr.sample(len(self.graph))

        def func(X, t=0):
            return self.get_terms(X)

        res = odeint(func, x0, t)
        return res

class LinearModel(ODEModel):
    info = {
        'name': 'Linear Model'
    }

    def setup(self):
        super(LinearModel, self).setup()

        self.f1 = lambda x: x
        self.f2 = lambda x: -x

class NonlinearModel(ODEModel):
    info = {
        'name': 'Nonlinear Model'
    }

    def setup(self):
        super(NonlinearModel, self).setup()

        self.f1 = lambda x: 1/(1+x)
        self.f2 = lambda x: x/(1+x)
