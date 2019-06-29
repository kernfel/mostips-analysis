# coding: utf-8

import fit_analysis as fa

groups = [{'names': ['DE', 'xGA', 'mGA']},
          {'names': ['cluster', 'bubble']},
          {'names': ['graded', 'target-only', 'unweighted']},
          {'names': ['cluster', 'bubble'], 'filt': 'DE'},
          {'names': ['graded', 'target-only', 'unweighted'], 'filt': ['DE', 'cluster']},
          {'names': ['graded', 'target-only', 'unweighted'], 'filt': ['DE', 'bubble']},
         ]

S = fa.Session('/home/kernfel/Documents/Data/RTDO/Kv14x_fixed/sessions/2019.06.18-15.56.02/',
               'full.index', 'Kv1.4x')

S.plot_all(figbase='fig_all_')

for g in groups:
    S.set_groups(**g)
    S.plot_all()