import os, os.path
import argparse, pprint

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
from matplotlib import rc
from matplotlib import ticker as ptk

from scipy.stats import gaussian_kde

import utils
from logger import log


# enable LaTeX
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

class Plotter(object):
    plot_save_directory = 'plot_dir'
    show_plots = True

    @staticmethod
    def show(name, fname=None, timestamp=False):
        """ Shows plot and automatically saves after closing preview
        """
        fname = os.path.join(Plotter.plot_save_directory, utils.clean_string(name)) if fname is None else fname

        # handle filesuffix
        if not (fname.endswith('.png') or
                fname.endswith('.svg')):
            fname += '.png'
        log('Plotting "%s"' % fname)

        # add timestamp
        if timestamp:
            parts = os.path.splitext(fname)
            no_ext = parts[0]
            no_ext += '_%s' % utils.get_strtime()
            fname = '%s%s' % (no_ext, parts[1])

        # handle surrounding directory structure
        dire = os.path.dirname(fname)
        if len(dire) > 0 and not os.path.exists(dire):
            os.makedirs(dire)

        # fix confusing axis offset
        formatter = ptk.ScalarFormatter(useOffset=False)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().yaxis.set_major_formatter(formatter)

        fig = plt.gcf()

        if Plotter.show_plots:
            plt.show()
        else:
            plt.close()

        # save plot
        fig.dpi = 250
        fig.savefig(fname, dpi=fig.dpi)

    @staticmethod
    def preprocess(**kwargs):
        """ Do some preprocessing if required
        """
        if 'axis_preprocessing' in kwargs:
            ax = plt.gca()

            cur = kwargs['axis_preprocessing']
            for k in cur:
                getattr(ax, k)(*cur[k][0], **cur[k][1])

    @staticmethod
    def plot_heatmap(data, title, xlabel, ylabel):
        """ Plot 2D array as heatmap
        """
        hm = plt.pcolor(data, cmap=cm.gray_r, vmin=0, vmax=1)

        plt.tick_params(labelsize=20)
        plt.axes().set_aspect('equal')

        plt.title(title, fontsize=33)
        plt.xlabel(xlabel, fontsize=30)
        plt.ylabel(ylabel, fontsize=30)

        Plotter.show(title)

    @staticmethod
    def multi_density_plot(data, title, xlabel, ylabel, border=0.2):
        """ Create density plot of given data
        """
        for e in data:
            cur = e['data']
            name = e['name']

            if (np.array(cur) == 0).all():
                continue

            density = gaussian_kde(cur)
            x = np.linspace(-border, border, len(cur))
            plt.plot(x, density(x), label=name)

        plt.tick_params(labelsize=20)

        plt.xlim(-border, border)

        plt.title(title, fontsize=33)
        plt.xlabel(xlabel, fontsize=30)
        plt.ylabel(ylabel, fontsize=30)

        #plt.legend(loc='best')

        Plotter.show(title)

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
    def plot_histogram(data, fname=None):
        plt.hist(data['data'], bins=data['args']['bins'] if 'bins' in data['args'] else 200)

        plt.title(data['title'])
        plt.xlabel(data['x_label'])
        plt.ylabel(data['y_label'])

        Plotter.show(data['title'] if fname is None else fname)

    @staticmethod
    def plot(data, fname=None):
        xs = []
        ys = []
        for i, j in zip(data['x_data'], data['y_data']):
            # remove 0-pairs
            if not (abs(i) < 1e-15 or abs(j) < 1e-15):
                xs.append(i)
                ys.append(j)

        plt.plot(
            xs, ys,
            linestyle='None',
            marker='.', markeredgecolor='blue',
            **(data['plt_args'] if 'plt_args' in data else {})
        )

        title = data['title']
        corr, p_val, = utils.StatsHandler.correlate(xs, ys)
        title += ' (corr: %.2f, p-value: %.2f)' % (round(corr, 2), round(p_val, 2))

        plt.title(title)
        plt.xlabel(data['x_label'])
        plt.ylabel(data['y_label'])

        Plotter.show(data['title'], fname=fname)

    @staticmethod
    def errorbar_plot(data, fname=None):
        """ This function assumes that y is a list of lists and automatically computes the error margin
        """
        def gen(dat):
            """ Generate mean and error terms for given data
            """
            y_mean = []
            y_err = []
            for e in dat:
                y_mean.append(np.mean(e))
                y_err.append(np.std(e))
            return y_mean, y_err

        if isinstance(data['y_data'][0][0], str):
            for lab, dat in data['y_data']:
                y_mean, y_err = gen(dat)
                plt.errorbar(data['x_data'], y_mean, yerr=y_err, fmt='o', label=lab)

            plt.legend(loc='best')
        else:
            y_mean, y_err = gen(data['y_data'])
            plt.errorbar(data['x_data'], y_mean, yerr=y_err, fmt='o')

            if 'polyfit' in data['args'] and data['args']['polyfit']:
                coeffs = np.polyfit(data['x_data'], y_mean, 1)

                y_vec = np.polyval(coeffs, data['x_data'])
                plt.plot(data['x_data'], y_vec, label='${0:.2}x{2} {1:.2}$'.format(coeffs[0], abs(coeffs[1]), '+' if coeffs[1] > 0 else '-'))

                plt.legend(loc='best')

        plt.title(data['title'])
        plt.xlabel(data['x_label'])
        plt.ylabel(data['y_label'])

        Plotter.show(data['title'], fname=fname)

    @staticmethod
    def multi_plot(data, fname=None):
        """ Plot multiple graphs into same coordinate system
        """
        for entry in data['data']:
            plt.plot(entry['x'], entry['y'], label=entry['label'])

        plt.title(data['title'])
        plt.xlabel(data['x_label'])
        plt.ylabel(data['y_label'])

        #plt.legend(loc='best')

        Plotter.show(data['title'], fname=fname)

    @staticmethod
    def loglog(data, show_corr=True, fname=None):
        """ Yield loglog plot
        """
        ax = Plotter.set_loglog(
            plt.gca(),
            data['x_data'], data['y_data'],
            data['title'],
            data['x_label'], data['y_label'],
            **(data['plt_args'] if 'plt_args' in data else {})
        )
        Plotter.show('%s.png' % data['title'], fname=fname)

    @staticmethod
    def multi_loglog(data, fname=None):
        """ Create subplot of loglog, where data['data'] is of the form [{x: <x>, y: <y>, ylabel: <yl>}]
        """
        fig, axarr = plt.subplots(len(data['data']), sharex=True)
        fig.suptitle(data['title'])

        for i, ax in enumerate(axarr):
            e = data['data'][i]
            Plotter.set_loglog(ax, e['x'], e['y'], ylabel=e['ylabel'])
        axarr[-1].set_xlabel(data['x_label'])

        Plotter.show('%s.png' % data['title'], fname=fname)

    @staticmethod
    def set_loglog(ax, x, y, title='', xlabel='', ylabel='', show_corr=True, **kwargs):
        """ Return loglog plot (axis) of given data and removes 0-pairs beforehand
        """
        xs = []
        ys = []
        for i, j in zip(x, y):
            # remove 0-pairs
            if not (abs(i) < 1e-15 or abs(j) < 1e-15):
                xs.append(i)
                ys.append(j)

        ax.loglog(
            xs, ys,
            linestyle='None',
            marker='.', markeredgecolor='blue',
            **kwargs
        )

        if show_corr:
            corr, p_val, = utils.StatsHandler.correlate(x, y)
            title += ' (corr: %.2f, p-value: %.2f)' % (round(corr, 2), round(p_val, 2))

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax


if __name__ == '__main__':
    """ Interactive plotting
    """
    # call `complete -r` to use path autocompletion
    parser = argparse.ArgumentParser(description='Interactive plotting of generated data')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-f',
        '--file',
        help='Use this data file to plot',
        type=str,
        metavar='<data file>'
    )
    group.add_argument(
        '-i',
        '--info',
        help='Show info of data file',
        type=str,
        metavar='<data file>',
        default=None
    )

    parser.add_argument(
        '-p',
        '--plot',
        help='Plot type to use instead of prefered one',
        type=str,
        metavar='<plot type>',
        default=None
    )
    parser.add_argument(
    	'-s',
    	'--save_only',
    	help='Only save and don\'t show result',
    	action='store_false'
    )
    parser.add_argument(
        '-o',
        '--output',
        help='File to save output to',
        type=str,
        metavar='<img file>',
        default=None
    )

    args = vars(parser.parse_args())

    if args['info'] is None:
        Plotter.show_plots = args['save_only']

        def get_name(info):
            if 'name' in info:
                if info['name'] == 'Boolean Model':
                    return '%(name)s_%(norm_time)s_%(cont_evo_runs)i_%(time_window)i' % info
            return None
        def handle_file(f):
            dic = utils.CacheHandler.load(f)
            func = getattr(Plotter, dic['info']['function'] if args['plot'] is None else args['plot'])

            Plotter.preprocess(**dic['args'])
            func(dic, fname=get_name(dic['info']) if args['output'] is None else args['output'])

        if os.path.isfile(args['file']):
            handle_file(args['file'])
        elif os.path.isdir(args['file']):
            for f in os.listdir(args['file']):
                fn = os.path.join(args['file'], f)
                if os.path.isfile(fn):
                    handle_file(fn)
                # don't recurse
        else:
            print('Could not find file, aborting')
    else:
        dic = utils.CacheHandler.load(args['info'])
        pprint.pprint(dic['info'])
