import numpy as np
import numpy.linalg as npl
import numpy.random as npr

from scipy.integrate import odeint


class Model(object):
    def __init__(self, graph):
        self.graph = graph
        self.setup()

    def setup(self):
        pass

    def generate(self):
        Exception('This model lacks any generating function')

    def get_augmented_adja_m(self):
        aug_adja_m = np.copy(self.graph.adja_m)

        # randomly inhibit or activate
        for y in range(aug_adja_m.shape[0]):
            for x in range(aug_adja_m.shape[1]):
                if aug_adja_m[x,y] == 1:
                    aug_adja_m[x,y] = npr.choice([-1, 1])

        # add self-inhibitory links for all nodes without incoming inhibitory links
        for x in range(aug_adja_m.shape[1]):
            col = np.copy(aug_adja_m[:,x])
            col[col==1] = 0
            s = sum(abs(col))

            if s == 0:
                aug_adja_m[x,x] = -1

        return aug_adja_m

class MultiplicatorModel(Model):
    """ Simulates network evolution by adjacency matrix multiplication
    """
    name = 'Multiplicator Model'

    def generate(self, runs=10):
        initial = np.array([npr.random() for i in range(len(self.graph))])

        data = [initial]
        for i in range(runs-1):
            cur = [self.graph.adja_m.dot(data[-1])[0, i] for i in range(len(self.graph))]
            cur /= npl.norm(cur)

            data = np.vstack((data, cur))

        return data

class BooleanModel(Model):
    """ Simple model to generate gene expression data
    """
    name = 'Boolean Model'

    def setup(self):
        """ Declare activating and inhibiting links and further constants
        """
        self.aug_adja_m = self.get_augmented_adja_m()

    def generate_binary_time_series(self, runs=10):
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

            for i in x.T:
                sigma = 0
                for j in x.T:
                    sigma += self.aug_adja_m[i,j] * j
                sigma = sigma.tolist()[0][0]

                res = -1
                if sigma > 0:
                    res = 1
                elif sigma < 0:
                    res = 0
                else:
                    res = i.tolist()[0][0]

                xnext.append(res)

            return xnext

        num = self.aug_adja_m.shape[0]
        x0 = npr.randint(2, size=num)

        data = np.matrix(x0)
        for t in range(runs-1):
            cur = rule(data[-1])
            data = np.vstack((data, cur))

        return np.array(data)

    def generate_continues_evolution(self, runs=10, norm_time=True):
        """ Generates continuous data from binary evolution
            Either norms along gene- or time-axis
            Returns:
            [
                [g1, g2, g3, ...], # at t0
                [g1, g2, g3, ...], # at t1
                ...
            ] # also format of "data", "concs"
        """
        time_window = 30
        model_runs = 300 # how many times to run the simulation

        data = np.array([])

        # concatenate time series
        for i in range(model_runs):
            res = self.generate_binary_time_series(runs)
            data = np.vstack((data, res)) if len(data) > 0 else res

        # average gene activations over time window
        concs = []
        for i in range(0, len(data), time_window):
            concs.append([])
            for n in range(len(self.graph)):
                window = data.T[n,i:i+time_window]
                concs[-1].append(window.mean())
        concs = np.array(concs)

        # normalize along genes/time
        out = []
        if norm_time:
            for time in concs.T:
                norm = npl.norm(time, 1)
                time /= norm if norm != 0 else 1
                out.append(time)
            out = np.array(out).T
        else:
            for genes in concs:
                norm = npl.norm(genes, 1)
                genes /= norm if norm != 0 else 1
                out.append(genes)
            out = np.array(out)

        return out

    def generate(self, runs=10):
        """ Averages many runs and returns result
        """
        avg_runs = 10

        tmp = None
        for i in range(avg_runs):
            cur = self.generate_continues_evolution(runs)
            if not tmp is None:
                tmp += cur
            else:
                tmp = cur

        return tmp / avg_runs

class ODEModel(Model):
    """ General model of ODEs to generate data
    """
    def setup(self):
        """ Setup needed constants
        """
        self.e1 = 0.2
        self.e2 = 0.2

        self.aug_adja_m = self.get_augmented_adja_m()

    def generate(self, runs=10):
        """ Solves nonlinear system and returns solution
        """
        num = self.aug_adja_m.shape[0]

        t = np.arange(0, runs, 1)
        x0 = npr.sample(num)

        def func(X, t=0):
            def iter(fun, i, fac):
                """ Generates term using given specified functions
                """
                sigma = 0
                for j in range(num):
                    e = self.aug_adja_m[i, j]
                    sigma += 0.5 * (abs(e) + fac * e) * fun(X[j])
                return sigma
            terms = []

            for i in range(num):
                terms.append(self.e1 * iter(self.f1, i, -1) + self.e1 * iter(self.f2, i, 1))

            return np.array(terms)

        res = odeint(func, x0, t)
        return res

class LinearModel(ODEModel):
    name = 'Linear Model'

    def setup(self):
        super(LinearModel, self).setup()

        self.f1 = lambda x: x
        self.f2 = lambda x: -x

class NonlinearModel(ODEModel):
    name = 'Nonlinear Model'

    def setup(self):
        super(NonlinearModel, self).setup()

        self.f1 = lambda x: 1/(1+x)
        self.f2 = lambda x: x/(1+x)
