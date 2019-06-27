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

def boxplot(data, group_names, param_names): # data: (groups, params, datapoints)
    numg, nump = len(group_names), len(param_names)

    for i, group in enumerate(data):
        line, = plt.plot([], label=group_names[i])
        col = line.get_color()
        b = plt.boxplot(group.T, positions=np.arange(i, (numg+1)*nump, (numg+1)),
                        widths=0.8, flierprops={'markeredgecolor':col})
        set_box_color(b, col)

    plt.xticks(np.arange((numg-1)/2., (numg+1)*nump, numg+1), param_names)
    plt.xlim(-1, (numg+1)*nump-1)
    plt.ylim(ymin=0)
    plt.legend()

class Session:
    def __init__(self, path, index_file, modelname):
        self.path = path
        self.modelname = modelname
        self.figbase = "figure_%f_%g"
        self.save_figures = True

        # Load index
        with open(path + '/' + index_file) as ifile:
            self.index = [row for row in csv.DictReader(ifile, delimiter = '\t')]

        # Extract group names
        self.gnames_raw = []
        for row in self.index:
            if row['group'] not in self.gnames_raw:
                self.gnames_raw.append(row['group'])
        self.set_groups(self.gnames_raw)

        # Extract cell names
        self.cnames = []
        for row in self.index:
            if row['cell'] not in self.cnames:
                self.cnames.append(row['cell'])

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
            with open(row['record'].replace('.atf', '.params')) as tfile:
                for trow in csv.reader(tfile, delimiter='\t'):
                    try:
                        idx = self.pnames.index(trow[0].split(':')[0])
                        row['reference'][idx] = float(trow[1].split(' ')[0])
                    except:
                        pass
        
        # Load population and validation data
        self.load_data()

        self.n_epochs = max(row['n_epochs'] for row in self.index)

    def load_data(self):
        for row in self.index:
            row['n_epochs'] = int(row['n_epochs'])
            row['n_pop'] = int(row['n_pop'])
            row['n_subpops'] = int(row['n_subpops'])
            row['subpop_sz'] = row['n_pop'] // row['n_subpops']
            
            fit = self.path + '/%04d.GAFitter.fit' % int(row['fileno'])
            
            # Load populations
            with open(fit + '.pops') as datafile:
                if int(row['subpop_split']):
                    row['data'] = np.fromfile(datafile, dtype=np.dtype(np.float32)).reshape(
                        (row['n_epochs'], self.nparams, 2, row['n_subpops'], row['subpop_sz']/2))\
                        .swapaxes(2,3).reshape(
                        (row['n_epochs'], self.nparams, row['n_subpops'], row['subpop_sz']))
                else:
                    row['data'] = np.fromfile(datafile, dtype=np.dtype(np.float32)).reshape(
                        (row['n_epochs'], self.nparams, row['n_subpops'], row['subpop_sz']))

            # Load step sizes
            with open(fit + '.steps') as datafile:
                # steptype: (params)
                # steps: (n_subpops, params, epochs)
                row['steptype'] = np.fromfile(datafile, dtype=np.dtype(np.int8), count = self.nparams)
                if np.any(row['steptype'] > 1):
                    assert np.all(row['steptype'] == row['n_subpops']), "DE multipopulation step type must equal n_subpops."
                    row['steps'] = np.fromfile(datafile, dtype=np.dtype(np.float32)).reshape(
                        (row['n_subpops'], self.nparams, row['n_epochs']))
                else:
                    steps = np.fromfile(datafile, dtype=np.dtype(np.float32)).reshape((self.nparams, row['n_epochs']))
                    row['steps'] = np.array([steps] * row['n_subpops'])

            # Load validations and crossvalidations
            failed = []
            for val_type in ['median_validation', 'lowerr_validation', 'median_xvalidation', 'lowerr_xvalidation']:
                try:
                    with open(fit + '.' + val_type) as datafile:
                        row[val_type] = np.fromfile(datafile, dtype=np.dtype(np.float64)).reshape(
                            (row['n_subpops'], row['n_epochs']))
                except:
                    if val_type not in failed:
                        print "Failed to open validation", val_type
                        failed.append(val_type)
            
            for val_type in ['target_validation', 'target_xvalidation']:
                try:
                    with open(fit + '.' + val_type) as datafile:
                        row[val_type] = np.fromfile(datafile, dtype=np.dtype(np.float64))
                except:
                    if val_type not in failed:
                        print "Failed to open validation", val_type
                        failed.append(val_type)

    def set_groups(self, names, filt = None):
        self.group_mapping = OrderedDict()
        for name in names:
            self.group_mapping[name] = [g for g in self.gnames_raw
                                        if name in g and
                                        (   (filt == None) or
                                            (type(filter) == str and filt in g) or
                                            (type(filter) == list and np.any([f in g for f in filt]))
                                        )]
        self.gnames = self.group_mapping.keys()
        self.filt = filt

    def index_by_cell(self, cell, rows=None):
        for row in rows or self.index:
            if row['cell'] == cell:
                yield row

    def index_by_group(self, gname, rows=None):
        for row in rows or self.index:
            if row['group'] in self.group_mapping[gname]:
                yield row

    def figname(self, name, fmt = 'svg'):
        if self.filt == None:
            fstr = ''
        elif type(self.filt) == str:
            fstr = self.filt + '_'
        else:
            fstr = '-'.join(self.filt) + '_'
        return self.path + '/' + self.figbase.replace('%g', '-'.join(self.gnames)).replace('%f_', fstr) + '_' + name + '.' + fmt

    def plot_all(self, save_figures = True, **kwargs):
        self.save_figures = save_figures

        self.plot_convergence(True, **kwargs)
        self.plot_convergence(False, **kwargs)
        plt.close('all')

        self.plot_spread(True, **kwargs)
        self.plot_spread(False, **kwargs)
        plt.close('all')

        self.plot_validation(True, True, **kwargs)
        self.plot_validation(False, True, **kwargs)
        self.plot_validation(True, False, **kwargs)
        self.plot_validation(False, False, **kwargs)
        plt.close('all')


############# Convergence ###################

    def plot_convergence(self, raw, **kwargs):
        if raw:
            title = self.modelname + ': Convergence (median absolute deviation from population median)'
            figname = 'raw-convergence'
        else:
            title = self.modelname + ': Convergence with reference (population medians\' absolute deviation from reference p.v.)'
            figname = 'ref-convergence'

        self.plot_convergence_box(raw, title, figname, **kwargs)

    def plot_convergence_grid(self, raw, title, figname, **kwargs):
        ''' return: (groups, fits, epochs, params, subpops) '''
        fig,ax = grid_setup(self.pnames, **kwargs)
        param_convergence = []
        lines = []
        pctile = kwargs.get('percentile', percentile)
        for gname in self.gnames:
            rows = self.index_by_group(gname)
            # data: (epochs, params, subpops, subpop_sz)
            # mads: (fits, epochs, params, subpops)
            # convergence: (epochs, params)
            if raw:
                mads = [np.median(np.abs(row['data'] - np.median(row['data'], axis=3)[:,:,:,None]), axis=3) for row in rows]
            else:
                mads = [np.abs(np.median(row['data'], axis=3) - row['reference'][None,:,None]) for row in rows]
            if len(mads) == 0:
                continue
            param_convergence.append(mads) # (groups, fits, epochs, params, subpops)
            convergence = np.percentile(mads, [100-pctile, 50, pctile], axis=(0,3))
            lines.append(plot_with_shade(ax, convergence))
        plt.figlegend(lines, self.gnames, 'upper right')
        plt.suptitle(title)
        figname = self.figname(figname)
        plt.savefig(figname)
        print figname

        return param_convergence

    def plot_convergence_norm(self, raw, title, figname, param_convergence = None, **kwargs):
        '''
        param_convergence: (groups, fits, epochs, params, subpops)
        output: (groups, fits, epochs, subpops)
        '''
        if param_convergence == None:
            param_convergence = self.plot_convergence_grid(raw, title, figname, **kwargs)

        fig,ax = fig_setup(xlabel='Epoch', ylabel='Parameter space distance (a.u.)', **kwargs)
        norm_convergence = [np.linalg.norm(np.divide(conv, self.sigmata[None,None,:,None]), axis=2) for conv in param_convergence]

        pctile = kwargs.get('percentile', percentile)
        lines = [plot_single_shaded(ax, np.percentile(distances, [100-pctile, 50, pctile], axis=(0,2))) for distances in norm_convergence]
        plt.figlegend(lines, self.gnames, 'upper right')
        plt.suptitle(title)
        figname = self.figname(figname + '-norm')
        plt.savefig(figname)
        print figname

        return norm_convergence

    def plot_convergence_box(self, raw, title, figname, norm_convergence = None, **kwargs):
        '''
        norm_convergence: (groups, fits, epochs, subpops)
        '''
        if norm_convergence == None:
            norm_convergence = self.plot_convergence_norm(raw, title, figname, **kwargs)

        fig_setup(ylabel='Parameter space distance (a.u.)', **kwargs)
        eps = kwargs.get('boxplot_epochs', boxplot_epochs)
        boxplot([np.moveaxis(conv[:,np.array(eps)-1,:], 1, 0).reshape(len(eps), -1)
                 for conv in norm_convergence],
                self.gnames, eps)
        plt.suptitle(title)
        figname = self.figname(figname + '-box')
        plt.savefig(figname)
        print figname


#################### Spread ######################

    def plot_spread(self, variance, **kwargs):
        if variance:
            title = self.modelname + ': Standard deviation of population medians'
            figname = 'spread-stddev'
        else:
            title = self.modelname + ': Range of population medians'
            figname = 'spread-range'
        self.plot_spread_grid(variance, title, figname, **kwargs)
        self.plot_spread_stepnorm_box(variance, title, figname, **kwargs)
        self.plot_spread_signorm_box(variance, title, figname, **kwargs)

    def plot_spread_grid(self, variance, title, figname, **kwargs):
        fig,ax = grid_setup(self.pnames, **kwargs)
        lines = []
        pctile = kwargs.get('percentile', percentile)
        for gname in self.gnames:
            group = [row for row in self.index_by_group(gname)]
            group_var = []
            for cname in self.cnames:
                rows = self.index_by_cell(cname, group)
                # data: (epochs, params, subpops, subpop_sz)
                medians = [np.median(row['data'], axis=3) for row in rows] # (fits, epochs, params, subpops)
                if len(medians) == 0:
                    continue
                if variance:
                    var = np.std(medians, axis=(0,3)) # (epochs, params)
                else:
                    var = np.max(medians, axis=(0,3)) - np.min(medians, axis=(0,3)) # (epochs, params)
                group_var.append(var) # (cells, epochs, params)
            spread = np.percentile(group_var, [100-pctile, 50, pctile], axis=0)
            lines.append(plot_with_shade(ax, spread))
        plt.figlegend(lines, self.gnames, 'upper right')
        plt.suptitle(title)
        figname = self.figname(figname)
        plt.savefig(figname)
        print figname

    def plot_spread_stepnorm_grid(self, variance, title, figname, **kwargs):
        ''' output: (groups, cells, epochs, params) '''
        fig,ax = grid_setup(self.pnames, **kwargs)
        lines = []
        pctile = kwargs.get('percentile', percentile)
        stepnorm_spread = []
        for gname in self.gnames:
            group = [row for row in self.index_by_group(gname)]
            group_var = []
            for cname in self.cnames:
                rows = [row for row in self.index_by_cell(cname, group)]
                if len(rows) == 0:
                    continue
                # data: (epochs, params, subpops, subpop_sz)
                # steps: (subpops, params, epochs)
                multiplicative_rows = filter(lambda r: np.any(r['steptype'] == 0), rows)
                for row in multiplicative_rows:
                    mask = row['steptype'] == 0
                    medians = np.median(row['data'], axis=3) # (epochs, params, subpops)
                    med_step = np.median(row['steps'], axis=0) # (params, epochs)
                    med_step[mask] = np.log(med_step[mask] + 1)
                    if variance:
                        mean = np.mean(medians, axis=2).T
                        std = np.std(medians, axis=2).T
                        std[mask] = np.log((mean[mask] + std[mask]) / mean[mask])
                        var = std / med_step
                    else:
                        min_median, max_median = np.min(medians, axis=2).T, np.max(medians, axis=2).T # (params, epochs)
                        min_median[mask] = np.log(min_median[mask])
                        max_median[mask] = np.log(max_median[mask])
                        med_step[mask] = np.log(med_step[mask] + 1)
                        var = (max_median - min_median) / med_step
                    group_var.append(var.T) # (cells, epochs, params)

                additive_rows = filter(lambda r: np.all(r['steptype'] > 0), rows)
                if len(additive_rows) > 0:
                    medians = [np.median(row['data'], axis=3) for row in additive_rows] # (fits, epochs, params, subpops)
                    med_step = np.median([row['steps'] for row in additive_rows], axis=(0,1)) # (params, epochs)
                    if variance:
                        var = np.std(medians, axis=(0,3))
                    else:
                        var = np.max(medians, axis=(0,3)) - np.min(medians, axis=(0,3)) # (epochs, params)
                    group_var.append(var / med_step.T) # (cells, epochs, params)




#                medians = [np.median(row['data'], axis=3) for row in rows] # (fits, epochs, params, subpops)
#                med_step = np.median([row['steps'] for row in rows], axis=(0,1)) # (params, epochs)

#                if np.any([row['steptype'] == 0 for row in rows]): # steptype==0 <=> multiplicative step size
#                    for row in rows:
#                        assert np.all((row['steptype']>0) == (rows[0]['steptype']>0)), 'Step type cannot differ within (group,cell) set.'
#                    mask = multiplicative_rows[0]['steptype'] == 0
#                    med_step[mask] = np.log(med_step[mask] + 1)
#                    if variance:
#                        mean = np.mean(medians, axis=(0,3)).T
#                        std = np.std(medians, axis=(0,3)).T
#                        std[mask] = np.log((mean[mask]+std[mask]) / mean[mask])
#                        var = std / med_step
#                    else:
#                        min_median, max_median = np.min(medians, axis=(0,3)).T, np.max(medians, axis=(0,3)).T # (params, epochs)
#                        min_median[mask] = np.log(min_median[mask])
#                        max_median[mask] = np.log(max_median[mask])
#                        med_step[mask] = np.log(med_step[mask] + 1)
#                        var = (max_median - min_median) / med_step
#                    group_var.append(var.T) # (cells, epochs, params)
#                else:
#                    if variance:
#                        var = np.std(medians, axis=(0,3))
#                    else:
#                        var = np.max(medians, axis=(0,3)) - np.min(medians, axis=(0,3)) # (epochs, params)
#                    group_var.append(var / med_step.T) # (cells, epochs, params)
            spread = np.percentile(group_var, [100-pctile, 50, pctile], axis=0)
            lines.append(plot_with_shade(ax, spread))
            stepnorm_spread.append(np.array(group_var)) # (groups, cells, epochs, params)
        plt.figlegend(lines, self.gnames, 'upper right')
        plt.suptitle(title + ', normalised by step size')
        figname = self.figname(figname + '-stepnorm')
        plt.savefig(figname)
        print figname

        return stepnorm_spread

    def plot_spread_stepnorm_box(self, variance, title, figname, stepnorm_spread = None, **kwargs):
        ''' stepnorm_spread: (groups, cells, epochs, params) '''
        if stepnorm_spread == None:
            stepnorm_spread = self.plot_spread_stepnorm_grid(variance, title, figname, **kwargs)

        for epoch in kwargs.get('boxplot_epochs', boxplot_epochs):
            fig_setup(ylabel = 'Stepsize-normalised spread (a.u.)', **kwargs)
            boxplot([cons[:,epoch-1,:].T for cons in stepnorm_spread], self.gnames, self.pnames)
            plt.suptitle(title + ', epoch %d' % epoch)
            fign = self.figname(figname + '-stepnorm-box-ep%d' % epoch)
            plt.savefig(fign)
            print fign

    def plot_spread_signorm(self, variance, title, figname, **kwargs):
        ''' output: (groups, cells, epochs) '''
        fig,ax = fig_setup(xlabel='Epoch', ylabel='Parameter space distance (a.u.)', **kwargs)
        lines = []
        pctile = kwargs.get('percentile', percentile)
        signorm_spread = []
        for gname in self.gnames:
            group = [row for row in self.index_by_group(gname)]
            group_var = []
            for cname in self.cnames:
                rows = [row for row in self.index_by_cell(cname, group)]
                if len(rows) == 0:
                    continue
                # data: (epochs, params, subpops, subpop_sz)
                # sigmata: (nparams,)
                medians = [np.median(row['data'], axis=3) for row in rows] # (fits, epochs, params, subpops)
                if variance:
                    var = np.std(medians, axis=(0,3))
                else:
                    var = np.max(medians, axis=(0,3)) - np.min(medians, axis=(0,3)) # (epochs, params)
                group_var.append(np.linalg.norm(var / self.sigmata, axis=1)) # (cells, epochs)
            spread = np.percentile(group_var, [100-pctile, 50, pctile], axis=0)
            lines.append(plot_single_shaded(ax, spread))
            signorm_spread.append(np.array(group_var)) # (groups, cells, epochs)
        plt.legend(lines, self.gnames)
        plt.suptitle(title)
        figname = self.figname(figname + '-signorm')
        plt.savefig(figname)
        print figname

        return signorm_spread

    def plot_spread_signorm_box(self, variance, title, figname, signorm_spread = None, **kwargs):
        ''' signorm_spread: (groups, cells, epochs) '''
        if signorm_spread == None:
            signorm_spread = self.plot_spread_signorm(variance, title, figname, **kwargs)

        fig_setup(ylabel='Parameter space distance (a.u.)', **kwargs)
        eps = kwargs.get('boxplot_epochs', boxplot_epochs)
        boxplot([cons[:,np.array(eps)-1].T for cons in signorm_spread], self.gnames, eps)
        plt.suptitle(title)
        figname = self.figname(figname + '-signorm-box')
        plt.savefig(figname)
        print figname


####################### Validation ###############################

    def plot_validation(self, median, cross, **kwargs):
        if median:
            src_key = 'median'
            src_title = 'population median models'
        else:
            src_key = 'lowerr'
            src_title = 'lowest fitting cost models'
        if cross:
            data_key = src_key + '_xvalidation'
            target_key = 'target_xvalidation'
            title = self.modelname + ': Cross-validation error (%s, all protocols)' % src_title
            figname = 'xvalidation-' + src_key
        else:
            data_key = src_key + '_validation'
            target_key = 'target_validation'
            title = self.modelname + ': Validation error (%s, target observations)' % src_title
            figname = 'validation-' + src_key

        fig,ax = fig_setup(xlabel='Epoch', ylabel='RMS current error (nA)', **kwargs)
        lines = []
        pctile = kwargs.get('percentile', percentile)
        # data: (subpop, epoch)
        for i, gname in enumerate(self.gnames):
            group = [row for row in self.index_by_group(gname)]
            val = np.concatenate([row[data_key] for row in group], axis=0)
            line = plot_single_shaded(ax, np.percentile(val, [100-pctile, 50, pctile], axis=0))
            lines.append(line)

            val_target = np.array([row[target_key] for row in group])
            box = plt.boxplot(val_target.reshape(-1,1), positions=(self.n_epochs + 10 + 10*i,), widths=10,
                              flierprops = {'markeredgecolor': line.get_color()}, manage_xticks = False)
            set_box_color(box, line.get_color())
        plt.xlim((-0.05*self.n_epochs, 1.05*(self.n_epochs+10+10*len(self.gnames))))
        plt.legend(lines, self.gnames)
        plt.suptitle(title)
        figname = self.figname(figname)
        plt.savefig(figname)
        print figname
