# coding: utf-8

import fit_analysis as fa

groups = [{'names': ['cluster', 'bubble']},
          {'names': ['graded', 'target-only', 'unweighted']},
          {'names': ['strategy 1', 'strategy 2', 'strategy 3'], 'filt': 'cluster'},
          {'names': ['strategy 1', 'strategy 2', 'strategy 3'], 'filt': 'bubble'}
         ]

S = fa.Session('/home/kernfel/Documents/Data/RTDO/Kx_both/sessions/2019.06.20-17.46.33/',
               'full.index', 'Kv2.1x + Kv1.4x')

kwargs = dict(rows = 4, cols = 2, figsize_grid = (15, 15))

S.plot_all(figbase='fig_all_', **kwargs)

for g in groups:
    S.set_groups(**g)
    S.plot_all(**kwargs)