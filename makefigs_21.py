# coding: utf-8

import fit_analysis as fa

groups = [{'names': ['DE', 'xGA', 'mGA'], 'filt': None},
          {'names': ['cluster', 'bubble'], 'filt': None},
          {'names': ['cluster', 'bubble'], 'filt': 'DE'},
          {'names': ['graded', 'target-only', 'unweighted'], 'filt': ['DE', 'cluster']},
          {'names': ['graded', 'target-only', 'unweighted'], 'filt': ['DE', 'bubble']}
         ]

S = fa.Session('/home/kernfel/Documents/Data/RTDO/Kv21x_fixed/sessions/2019.06.19-12.16.08/',
               'full.index', 'Kv2.1x')

for g in groups:
    S.set_groups(**g)
    S.plot_all()