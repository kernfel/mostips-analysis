# coding: utf-8
# post-hoc analysis of RTDO fit outputs

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from collections import OrderedDict

figsize_single = (10,5)
figsize_grid = (15,10)
rows, cols = 3,2

percentile = 25
boxplot_epochs = (100,250,500)

def plot_single_shaded(ax, triplet):
    '''
    Draws a triplet of lines on ax, where triplet[(0,2),:] are the borders (e.g. upper/lower quartiles),
    and triplet[1,:] is the central line (e.g. median).
    Returns: The central line's Line2D object
    '''
    baseline, = ax.plot(triplet[1,:])
    color = baseline.get_color()
    ax.fill_between(range(len(triplet[0,:])), triplet[0,:], triplet[2,:],
                    facecolor=color+'55', edgecolor=color+'99')
    return baseline

def plot_with_shade(axes, triplets):
    '''
    Draws a set of shaded lines (see plot_single_shaded), each on its respective axis.
    Triplets should be of shape (3, x, nparams). At least nparams axes must be passed.
    Returns: the last triplet's central line
    '''
    for i in range(triplets.shape[2]):
        baseline = plot_single_shaded(axes.ravel()[i], triplets[:,:,i])
    return baseline

def fig_setup(xlabel = '', ylabel = '', **kwargs):
    fig,ax = plt.subplots(figsize=kwargs.get('figsize_single', figsize_single))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    return fig,ax

def grid_setup(pnames, xlabel = 'Epoch', **kwargs):
    fig,axes = plt.subplots(kwargs.get('rows', rows), kwargs.get('cols', cols), sharex='col', figsize=kwargs.get('figsize_grid', figsize_grid))
    for i,ax in enumerate(axes.ravel()):
        if i < len(pnames):
            ax.set_ylabel(pnames[i])
            if i >= len(pnames) - cols:
                ax.set_xlabel(xlabel)
    return fig,axes

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def boxplot(data, group_names, param_names): # data: (groups, datapoints, params)
    numg, nump = len(group_names), len(param_names)

    for i, group in enumerate(data):
        line, = plt.plot([], label=group_names[i])
        col = line.get_color()
        b = plt.boxplot(group, positions=np.arange(i, (numg+1)*nump, (numg+1)),
                        widths=0.8, flierprops={'markeredgecolor':col})
        set_box_color(b, col)

    plt.xticks(np.arange((numg-1)/2., (numg+1)*nump, numg+1), param_names)
    plt.xlim(-1, (numg+1)*nump-1)
    plt.legend()

class Session:
    def __init__(self, path, index_file, modelname, load = True):
        self.path = path
        self.modelname = modelname
        self.figbase = "figure_%f_%g"

        # Load index
        with open(path + '/' + index_file) as ifile:
            self.index = [row for row in csv.DictReader(ifile, delimiter = '\t')]

        # Extract names
        self.gnames_raw = []
        self.cnames = []
        self.recnames = []
        for row in self.index:
            if row['group'] not in self.gnames_raw:
                self.gnames_raw.append(row['group'])
            if row['cell'] not in self.cnames:
                self.cnames.append(row['cell'])
            if row['record'] not in self.recnames:
                self.recnames.append(row['record'])

        self.set_groups(self.gnames_raw)
        self.exclude_cells(None)

        # Load param names and deltabar-normalised sigmata
        with open(path + '/../../paramnames') as pfile:
            self.pnames, sigmata = zip(*[(row[0], float(row[1])) for row in csv.reader(pfile, delimiter = '\t')])
            with open(path + '/../../deltabar') as dfile:
                deltabar = [float(row[0]) for row in csv.reader(dfile)]
            self.sigmata = np.divide(sigmata, deltabar)
        self.nparams = len(self.pnames)

        # Load target parameter values
        for row in self.index:
            row['reference'] = np.zeros(self.nparams)
            if  row['record'] == 'simulated':
                with open(path + '/' + row['cell'] + '.params') as tfile:
                    row['reference'] = np.array([float(p[0]) for p in csv.reader(tfile)])
            else:
                with open(row['record'].replace('.atf', '.params')) as tfile:
                    for trow in csv.reader(tfile, delimiter='\t'):
                        try:
                            idx = self.pnames.index(trow[0].split(':')[0])
                            row['reference'][idx] = float(trow[1].split(' ')[0])
                        except:
                            pass

        # Ensure correct typing
        for row in self.index:
            row['fileno'] = int(row['fileno'])
            row['n_epochs'] = int(row['n_epochs'])
            row['n_pop'] = int(row['n_pop'])
            row['n_subpops'] = int(row['n_subpops'])
            row['subpop_sz'] = row['n_pop'] // row['n_subpops']
        self.n_epochs = min(row['n_epochs'] for row in self.index)

        # Load population and validation data
        if load:
            self.load_populations()
            self.load_validations()

    def load_populations(self):
        for i,row in enumerate(self.index):
            print("Loading row %d of %d, fileno %04d..." % (i, len(self.index), row['fileno']))

            fit = self.path + '/%04d.GAFitter.fit' % row['fileno']

            try:
                # Load populations : (n_epochs, nparams, n_subpops, subpop_sz)
                with open(fit + '.pops') as datafile:
                    if int(row['subpop_split']):
                        pops = np.fromfile(datafile, dtype=np.dtype(np.float32)).reshape(
                            (row['n_epochs'], self.nparams, 2, row['n_subpops'], row['subpop_sz']/2))\
                            .swapaxes(2,3).reshape(
                            (row['n_epochs'], self.nparams, row['n_subpops'], row['subpop_sz']))
                    else:
                        pops = np.fromfile(datafile, dtype=np.dtype(np.float32)).reshape(
                            (row['n_epochs'], self.nparams, row['n_subpops'], row['subpop_sz']))

                # Load errors to determine lowest-cost models : (n_epochs, n_subpops, subpop_sz)
                with open(fit + '.errs') as errfile:
                    if int(row['subpop_split']):
                        errs = np.fromfile(errfile, dtype=np.dtype(np.float32)).reshape(
                            (row['n_epochs'], 2, row['n_subpops'], row['subpop_sz']/2))\
                            .swapaxes(1,2).reshape(
                            (row['n_epochs'], row['n_subpops'], row['subpop_sz']))
                    else:
                        errs = np.fromfile(errfile, dtype=np.dtype(np.float32)).reshape(
                            (row['n_epochs'], row['n_subpops'], row['subpop_sz']))

                # row data (lowerr, median, lowerr_mad, median_mad, std): (n_epochs, nparams, n_subpops)
                indices = np.argmin(errs, axis=2)
                row['lowerr'] = pops[np.ogrid[:row['n_epochs'], :self.nparams, :row['n_subpops']] + [indices[:,None,:]]]

                row['median'] = np.median(pops, axis=3)

                row['lowerr_mad'] = np.median(np.abs(pops - row['lowerr'][:,:,:,None]), axis=3)
                row['median_mad'] = np.median(np.abs(pops - row['median'][:,:,:,None]), axis=3)

                row['std'] = np.std(pops, axis=3)

            except IOError:
                for set in ['lowerr', 'median', 'lowerr_mad', 'median_mad', 'std']:
                    row[set] = np.fromfile(fit + '.' + set, dtype=np.dtype(np.float32)).reshape(row['n_epochs'], self.nparams, row['n_subpops'])

    def load_validations(self):
        for row in self.index:
            fit = self.path + '/%04d.GAFitter.fit' % int(row['fileno'])

            for val_type in ['median_validation', 'lowerr_validation', 'median_xvalidation', 'lowerr_xvalidation']:
                try:
                    with open(fit + '.' + val_type) as datafile:
                        row[val_type] = np.fromfile(datafile, dtype=np.dtype(np.float64)).reshape(
                            (row['n_subpops'], row['n_epochs']))
                except:
                    print "Failed to open %s.%s" % (fit,val_type)

            for val_type in ['target_validation', 'target_xvalidation']:
                try:
                    with open(fit + '.' + val_type) as datafile:
                        row[val_type] = np.fromfile(datafile, dtype=np.dtype(np.float64))
                except:
                    print "Failed to open %s.%s" % (fit,val_type)

    def compress(self, destructive = False):
        for row in self.index:
            fit = self.path + '/%04d.GAFitter.fit' % row['fileno']
            for set in ['lowerr', 'median', 'lowerr_mad', 'median_mad', 'std']:
                row[set].tofile(fit + '.' + set)
            if destructive:
                for set in ['pops', 'errs']:
                    try:
                        os.remove(fit + '.' + set)
                    except OSError as e:
                        print(e)

    def set_groups(self, names, filt = None, strict_filter = True):
        self.group_mapping = OrderedDict()
        for name in names:
            self.group_mapping[name] = [g for g in self.gnames_raw
                                        if name in g and
                                        (   (filt == None) or
                                            (type(filt) == str and filt in g) or
                                            (type(filt) == list and strict_filter and np.all([f in g for f in filt])) or
                                            (type(filt) == list and not strict_filter and np.any([f in g for f in filt]))
                                        )]
        self.gnames = self.group_mapping.keys()
        self.filt = filt

    def exclude_cells(self, cellnames):
        ''' Exclude the given cells from all analysis '''
        if type(cellnames) == str and len(cellnames) > 0:
            cellnames = [cellnames]
        elif type(cellnames) != list:
            cellnames = []
        self.excluded_cells = cellnames


    def index_by_cell(self, cell, rows=None):
        for row in rows or self.index:
            if row['cell'] == cell and cell not in self.excluded_cells:
                yield row

    def index_by_group(self, gname, rows=None):
        for row in rows or self.index:
            if row['group'] in self.group_mapping[gname] and row['cell'] not in self.excluded_cells:
                yield row

    def figname(self, name, fmt = 'svg'):
        if self.filt == None:
            fstr = ''
        elif type(self.filt) == str:
            fstr = self.filt + '_'
        else:
            fstr = '-'.join(self.filt) + '_'
        return self.path + '/' + self.figbase.replace('%g', '-'.join(self.gnames)).replace('%f_', fstr) + '_' + name + '.' + fmt

    def get_legend_n(self):
        legend = []
        for gname in self.gnames:
            gdata = [row for row in self.index_by_group(gname)]
            n = np.sum([row['n_subpops'] for row in gdata])
            nc = len(np.unique([row['cell'] for row in gdata]))
            nr = len(np.unique([row['record'] for row in gdata]))
            legend.append('%s (n=%d fits, %d cells, %d recs)' % (gname, n, nc, nr))
        return legend

    def plot_all(self, figbase = None, **kwargs):
        if figbase:
            figbase_bk = self.figbase
            self.figbase = figbase

        self.plot_all_convergence(**kwargs)
        self.plot_all_validation(**kwargs)

        if figbase:
            self.figbase = figbase_bk


############# Convergence ###################

    def plot_all_convergence(self, **kwargs):
        self.plot_convergence('popstd', **kwargs)
        for reftype in ['popmad', 'cell', 'record', 'external']:
            for center in ['median', 'lowerr']:
                self.plot_convergence(reftype, center, **kwargs)

    def plot_convergence(self, reftype, center = 'median', **kwargs):
        ref_title = dict(   popstd = 'within populations (stddev)',
                            popmad = 'within populations (MAD)',
                            cell = 'across fits to the same cell',
                            record = 'across fits to the same data',
                            external = 'with classical fit'
                        )
        assert ref_title.has_key(reftype), 'Valid reftypes: ' + ', '.join(ref_title.keys())
        assert center in ['median', 'lowerr']
        figname = 'convergence_' + center + '_' + reftype
        if reftype == 'popstd':
            conv = ': Convergence '
            figname = 'convergence_' + reftype
        elif center == 'median':
            conv = ': Median model convergence '
        elif center == 'lowerr':
            conv = ': Best-fit model convergence '
        title = self.modelname + conv + ref_title[reftype]

        data = self.get_convergence(reftype, center) # (groups, fits, epochs, params, subpops)
        abs_data = [np.abs(group) for group in data]
        norm_data = [np.linalg.norm(np.divide(group, self.sigmata[None,None,:,None]), axis=2) for group in data] # (groups, fits, epochs, subpops)

        if reftype in ['cell', 'record', 'external']:
            self.plot_grid(title + ' (abs)', figname + '-abs', abs_data, **kwargs)
        if reftype in ['popmad', 'popstd', 'external']:
            self.plot_grid(title, figname, data, **kwargs)
        self.plot_norm(title, figname, norm_data, **kwargs)
        self.plot_boxes(title, figname, norm_data, **kwargs)
        plt.close('all')

    def get_convergence(self, reftype, center):
        ''' output: (groups, fits, epochs, params, subpops) '''
        convergence = []
        for gname in self.gnames:
            gdata = [row for row in self.index_by_group(gname)]

            # row['*_mad'/'std'/'median'/'lowerr']: (epochs, params, subpops)
            # row['reference']: (params)
            # group_convergence: (fits, epochs, params, subpops)

            if reftype == 'popmad':
                group_convergence = [row[center + '_mad'] for row in gdata]
            elif reftype == 'popstd':
                group_convergence = [row['std'] for row in gdata]
            elif reftype == 'cell':
                ref = dict()
                g_cnames = [row['cell'] for row in gdata]
                for cname in self.cnames:
                    if cname in g_cnames:
                        ref[cname] = np.median([row[center] for row in gdata if row['cell'] == cname], axis=(0,3)) # (epochs, params)
                group_convergence = [ref[row['cell']][:,:,None] - row[center] for row in gdata]
            elif reftype == 'record':
                ref = dict()
                g_recnames = [row['record'] for row in gdata]
                for recname in self.recnames:
                    if recname in g_recnames:
                        ref[recname] = np.median([row['median'] for row in gdata if row['record'] == recname], axis=(0,3)) # (epochs, params)
                group_convergence = [ref[row['record']][:,:,None] - row[center] for row in gdata]
            elif reftype == 'external':
                group_convergence = [row['reference'][None,:,None] - row[center] for row in gdata]

            convergence.append(group_convergence)
        return convergence

    def plot_grid(self, title, figname, data, **kwargs):
        ''' data: (groups, fits, epochs, params, subpops) '''
        fig,ax = grid_setup(self.pnames, **kwargs)
        pctile = kwargs.get('percentile', percentile)
        pctiles = [np.percentile(group, [100-pctile, 50, pctile], axis=(0,3)) for group in data]
        lines = [plot_with_shade(ax, group) for group in pctiles]
        plt.figlegend(lines, self.get_legend_n(), 'upper right');
        plt.suptitle(title)
        f = self.figname(figname)
        plt.savefig(f)
        plt.close(fig)
        print f

    def plot_norm(self, title, figname, norm_data, **kwargs):
        ''' norm_data: (groups, fits, epochs, subpops) '''
        fig,ax = fig_setup(xlabel='Epoch', ylabel='Parameter space distance (a.u.)', **kwargs)
        pctile = kwargs.get('percentile', percentile)
        pctiles = [np.percentile(group, [100-pctile, 50, pctile], axis=(0,2)) for group in norm_data]
        lines = [plot_single_shaded(ax, group) for group in pctiles]
        plt.figlegend(lines, self.get_legend_n(), 'upper right')
        plt.suptitle(title)
        f = self.figname(figname + '-norm')
        plt.savefig(f)
        plt.close(fig)
        print f

    def plot_boxes(self, title, figname, norm_data, **kwargs):
        ''' norm_data: (groups, fits, epochs, subpops) '''
        fig,ax = fig_setup(ylabel='Parameter space distance (a.u.)', **kwargs)
        eps = kwargs.get('boxplot_epochs', boxplot_epochs)
        boxplot([np.moveaxis(group[:,np.array(eps)-1,:], 1, 2).reshape(-1, len(eps))
                 for group in norm_data],
                self.get_legend_n(), eps)
        plt.ylim(ymin=0)
        plt.suptitle(title)
        f = self.figname(figname + '-box')
        plt.savefig(f)
        plt.close(fig)
        print f


####################### Validation ###############################

    def plot_all_validation(self, **kwargs):
        for median in [True, False]:
            for cross in [True, False]:
                for norm in [True, False]:
                    self.plot_validation(median, cross, norm, **kwargs)

    def plot_validation(self, median, cross, norm, **kwargs):
        if median:
            src_key = 'median'
            src_title = 'median models'
        else:
            src_key = 'lowerr'
            src_title = 'best-fit models'

        if norm:
            norm_key = 'lognorm'
            norm_title = 'log norm error relative to classical fit'
            ylabel = 'log norm error ($\ln(\epsilon/\epsilon_0)$)'
        else:
            norm_key = 'err'
            norm_title = 'error'
            ylabel = 'RMS current error (nA)'

        if cross:
            val_key = 'xvalidation'
            title = self.modelname + ': Cross-validation %s (%s, all data)' % (norm_title, src_title)
        else:
            val_key = 'validation'
            title = self.modelname + ': Validation %s (%s, target observations)' % (norm_title, src_title)

        data_key = src_key + '_' + val_key
        target_key = 'target_' + val_key
        figname = val_key + '-' + src_key + '-' + norm_key

        ## Time series
        fig,ax = fig_setup(xlabel='Epoch', ylabel=ylabel, **kwargs)
        lines = []
        pctile = kwargs.get('percentile', percentile)
        validation = []
        validation_target = []
        for i, gname in enumerate(self.gnames):
            group = [row for row in self.index_by_group(gname)]

            # row[data_key]: (subpop, epoch)
            # row[target_key]: single value
            # val: (pop, epoch)

            if norm:
                val = np.log(np.concatenate([row[data_key] / row[target_key] for row in group], axis=0))
            else:
                val = np.concatenate([row[data_key] for row in group], axis=0)

            line = plot_single_shaded(ax, np.percentile(val, [100-pctile, 50, pctile], axis=0))
            lines.append(line)
            validation.append(val) # (group, pop, epoch)

            if not norm:
                val_target = np.array([row[target_key] for row in group])
                validation_target.append(val_target) # (group, fit)
                box = plt.boxplot(val_target.reshape(-1,1), positions=(self.n_epochs + 10 + 10*i,), widths=10,
                                  flierprops = {'markeredgecolor': line.get_color()}, manage_xticks = False)
                set_box_color(box, line.get_color())

        if norm:
            plt.axhline(0, color = 'gray', ls = 'dotted')
        else:
            plt.xlim((-0.05*self.n_epochs, 1.05*(self.n_epochs+10+10*len(self.gnames))))
        plt.legend(lines, self.get_legend_n())
        plt.suptitle(title)
        f = self.figname(figname)
        plt.savefig(f)
        plt.close(fig)
        print f

        ## Boxes
        fig,ax = fig_setup(ylabel=ylabel, **kwargs)
        eps = kwargs.get('boxplot_epochs', boxplot_epochs)
        selected_validation = [val[:,np.array(eps)-1] for val in validation]
        if norm:
            plt.axhline(0, color = 'gray', ls = 'dotted')
            boxplot(selected_validation, self.get_legend_n(), eps)
        else:
            val_all = [sel.T.tolist() + tar[None,:].tolist() for sel,tar in zip(selected_validation, validation_target)]
            boxplot(val_all, self.get_legend_n(), list(eps) + ['classical fit'])
        plt.suptitle(title)
        f = self.figname(figname + '-box')
        plt.savefig(f)
        plt.close(fig)
        print f
