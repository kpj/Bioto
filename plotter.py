import os, os.path

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
from matplotlib import rc

import utils


# enable LaTeX
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

class Plotter(object):
    plot_save_directory = 'plot_dir'
    show_plots = True

    @staticmethod
    def show(name):
        """ Shows plot and automatically saves after closing preview
        """
        fig = plt.gcf()

        if Plotter.show_plots:
            plt.show()
        else:
            plt.close()

        if not os.path.exists(Plotter.plot_save_directory):
            os.makedirs(Plotter.plot_save_directory)
        fig.savefig(os.path.join(Plotter.plot_save_directory, name.replace(' ', '_')), dpi=150)

    @staticmethod
    def present_graph(data, perron_frobenius, page_rank, degree_distribution):
        """ Shows a nice representation of the graphs features after evolution
        """
        info = [
            {
                'data': data[::-1],
                'title': 'Excitation Development via Adjacency-Matrix Multiplication',
                'rel_height': 6
            },
            {
                'data': np.array([perron_frobenius]),
                'title': 'Perron-Frobenius Eigenvector',
                'rel_height': 1
            }#,
#            {
#                'data': np.array([page_rank]),
#                'title': 'Pagerank',
#                'rel_height': 1
#            },
#            {
#                'data': np.array([degree_distribution]),
#                'title': 'Degree Distribution',
#                'rel_height': 1
#            }
        ]

        dimen = 20

        hm = plt.subplot2grid((dimen,dimen), (0,0), rowspan=17, colspan=19)
        pf = plt.subplot2grid((dimen,dimen), (18,0), rowspan=5, colspan=19)
        cb = plt.subplot2grid((dimen,dimen), (0,19), rowspan=20)

        for entry, ax in zip(info, [hm, pf]):
            hm = ax.pcolor(entry['data'], cmap=cm.gray, vmin=0, vmax=1)

            ax.set_title(entry['title'], fontsize=23)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        pcb = plt.colorbar(hm, cax=cb)
        pcb.ax.tick_params(labelsize=20)

        """hm = plt.subplot2grid((3,3), (0,0), rowspan=2, colspan=2)
        pf = plt.subplot2grid((3,3), (2,0), colspan=2)
        cb = plt.subplot2grid((3,3), (0, 2), rowspan=3)

        h = hm.pcolor(data[::-1], cmap=cm.gray, vmin=-0.1, vmax=1)
        pf.pcolor(np.array([perron_frobenius]), cmap=cm.gray, vmin=-0.1, vmax=1)
        plt.colorbar(h, cax=cb)"""

        Plotter.show('overview')

    @staticmethod
    def plot(x, y, title, xlabel='', ylabel='', show_corr=True):
        plt.plot(
            x, y,
            linestyle='None',
            marker='.', markeredgecolor='blue'
        )

        orig_title = title
        if show_corr:
            corr, p_val, = utils.StatsHandler.correlate(x, y)
            title += ' (corr: %.2f, p-value: %.2f)' % (round(corr, 2), round(p_val, 2))

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        Plotter.show(orig_title)

    @staticmethod
    def errorbar_plot(x, y, title='', xlabel='', ylabel=''):
        """ This function assumes that y is a list of lists and automatically computes the error margin
        """
        y_mean = []
        y_err = []
        for e in y:
            y_mean.append(np.mean(e))
            y_err.append(np.std(e))

        plt.errorbar(x, y_mean, yerr=y_err, fmt='o')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        Plotter.show(title)

    @staticmethod
    def multi_plot(data, title='', xlabel='', ylabel=''):
        """ Plots multiple graphs into same coordinate system
        """
        for entry in data:
            plt.plot(entry['x'], entry['y'], label=entry['label'])

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        #plt.legend(loc='best')

        Plotter.show(title)

    @staticmethod
    def loglog(x, y, title, xlabel='', ylabel=''):
        """ Yields loglog plot
        """
        ax = Plotter.set_loglog(plt.gca(), x, y, title, xlabel, ylabel)
        Plotter.show('%s.png' % title)

    @staticmethod
    def multi_loglog(title, xlabel, data):
        """ Creates subplot of loglog, where data is of the form [{x: <x>, y: <y>, ylabel: <yl>}]
        """
        fig, axarr = plt.subplots(len(data), sharex=True)
        fig.suptitle(title)

        for i, ax in enumerate(axarr):
            e = data[i]
            Plotter.set_loglog(ax, e['x'], e['y'], ylabel=e['ylabel'])
        axarr[-1].set_xlabel(xlabel)

        Plotter.show('%s.png' % title)

    @staticmethod
    def set_loglog(ax, x, y, title='', xlabel='', ylabel='', show_corr=True):
        """ Returns loglog plot (axis) of given data and removes 0-pairs beforehand
        """
        xs = []
        ys = []
        for i, j in zip(x, y):
            # remove 0-pairs
            if not (i < 1e-15 or j < 1e-15):
                xs.append(i)
                ys.append(j)

        ax.loglog(
            xs, ys,
            linestyle='None',
            marker='.', markeredgecolor='blue'
        )

        if show_corr:
            corr, p_val, = utils.StatsHandler.correlate(x, y)
            title += ' (corr: %.2f, p-value: %.2f)' % (round(corr, 2), round(p_val, 2))

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax
