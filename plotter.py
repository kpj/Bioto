import os, os.path

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec


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
            },
            {
                'data': np.array([page_rank]),
                'title': 'Pagerank',
                'rel_height': 1
            },
            {
                'data': np.array([degree_distribution]),
                'title': 'Degree Distribution',
                'rel_height': 1
            }
        ]

        gs = gridspec.GridSpec(len(info), 1, height_ratios=[e['rel_height'] for e in info])
        for entry, g in zip(info, gs):
            ax = plt.subplot(g)

            ax.pcolor(entry['data'], cmap=cm.gray, vmin=-0.1, vmax=1)

            ax.set_title(entry['title'])
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        Plotter.show('overview')

    @staticmethod
    def multi_plot(title, data):
        """ Plots multiple graphs into same coordinate system
        """
        for entry in data:
            plt.plot(entry['x'], entry['y'], label=entry['label'])
        plt.legend(loc='best')

        Plotter.show(title)

    @staticmethod
    def loglog(x, y, title='', xlabel='', ylabel=''):
        """ Yields loglog plot
        """
        ax = Plotter.set_loglog(plt.gca(), x, y, title, xlabel, ylabel)
        Plotter.show('%s.png')

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

        Plotter.show('%s.png')

    @staticmethod
    def set_loglog(ax, x, y, title='', xlabel='', ylabel=''):
        """ Returns loglog plot (axis) of given data and removes 0-pairs beforehand
        """
        xs = []
        ys = []
        for i, j in zip(x, y):
            # remove 0-pairs
            if not (i == 0 or j == 0):
                xs.append(i)
                ys.append(j)

        ax.loglog(
            xs, ys,
            linestyle='None',
            marker='.', markeredgecolor='blue'
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax
