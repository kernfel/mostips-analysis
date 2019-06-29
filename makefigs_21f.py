# coding: utf-8

import fit_analysis as fa

groups = [{'names': ['DE', 'xGA', 'mGA']},
          {'names': ['cluster', 'bubble']},
          {'names': ['cluster', 'bubble'], 'filt': 'DE'},
          {'names': ['cluster', 'bubble'], 'filt': 'xGA'},
          {'names': ['cluster', 'bubble'], 'filt': 'mGA'},
          {'names': ['strategy 1', 'strategy 2', 'strategy 3'], 'filt': ['DE', 'cluster']},
          {'names': ['strategy 1', 'strategy 2', 'strategy 3'], 'filt': ['DE', 'bubble']}
         ]

S = fa.Session('/home/kernfel/Documents/Data/RTDO/Kv21x_fixed/sessions/2019.06.19-12.16.08/',
               'full.index', 'Kv2.1x (fixed kinetics)')

S.plot_all(figbase='fig_all_')

for g in groups:
    S.set_groups(**g)
    S.plot_all()