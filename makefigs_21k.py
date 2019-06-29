# coding: utf-8

import fit_analysis as fa

groups = [{'names': ['DE', 'xGA', 'mGA']},
          {'names': ['cluster', 'bubble']},
          {'names': ['cluster', 'bubble'], 'filt': 'DE'}
         ]

S = fa.Session('/home/kernfel/Documents/Data/RTDO/Kv21x_kinetics/sessions/2019.06.20-12.20.10/',
               'full.index', 'Kv2.1x (incl. kinetics)')

S.plot_all(figbase='fig_all_')

for g in groups:
    S.set_groups(**g)
    S.plot_all()