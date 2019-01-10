# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import stfio
from analysis_helpers import *

# Time of one sample in ms
tres = 0.025

# The start and end of the activation step, with some room for the capacitive current to dissipate:
rec_limits = (5200, 44800)

# The true step time
rec_step_t0 = 4937

# The start and end of the tail step -- here, the first number is the true step time
rec2_limits = (4750, 40000)
rec2_prepulse = 751

# Index of the first capacitance step
rec3_offset = 322

# Capacitance step spec
rec3_stepdur = 2000
rec3_nsteps = 9

class Analysis:
    def __init__(self, filebase, filenos, factor = 1):
        self.savebase = filebase[:-4] % filenos[0] + '-' + str(filenos[2] or filenos[1])
        self.paramsfile = filebase[:-4] % filenos[3] + '.params'
        self.params = dict()
        
        self.rec = read_2channel_ATF(filebase % filenos[0], current_factor = factor)
        self.rec2 = read_2channel_ATF(filebase % filenos[1], current_factor = factor)
        self.rec3 = read_2channel_ATF(filebase % filenos[2]) if filenos[2] else None
    
    def fit(self):
        self.fit_leak()
        self.fit_EK()
        self.fit_gK()
        self.fit_C()
    
    def fit_leak(self):
        fit_leak(self.rec, self.params, None, rec_limits)
    
    def fit_EK(self):
        fit_tails_exp2(self.rec2, rec2_limits[0], rec2_limits[1])
        
        tail_t0 = [exp2_decay(0, self.rec2.pdecay[i]) for i in range(len(self.rec2.tails))]

        included = linear_exclude_outliers(self.rec2.tail_voltages, tail_t0)
        tail_I = np.array(tail_t0)[included]
        tail_V = np.array(self.rec2.tail_voltages)[included]

        # Fit to the positive current values only
        tail_fit_from = 0
        for i in range(len(tail_I)-1, 0, -1):
            if tail_I[i] < 0:
                tail_fit_from = i+1
                break

        fit_IV(tail_I[tail_fit_from:], tail_V[tail_fit_from:], self.params, 'K')
        
    def fit_gK(self):
        median_voltages = [np.median(V[rec_limits[0]:rec_limits[1]]) for V in self.rec.voltage]
        peak_currents = [np.max(I[rec_limits[0]:rec_limits[1]]) - self.params['I_leak'](V)
                         for I,V in zip(self.rec.current, median_voltages)]

        self.params['g_K'] = 1.05*peak_currents[-1] / (median_voltages[-1] - self.params['E_K'])
    
    def fit_C(self):
        if self.rec3:
            self.params['C'] = fit_capacitance(self.rec3, tres, rec3_offset, rec3_stepdur, rec3_nsteps)
        else:
            self.params['C'] = fit_capacitance_rec(self.rec, tres, rec_step_t0, rec3_stepdur)
            
    def params_str(self):
        string = '\
gl:\t%(g_leak)f μS\n\
El:\t%(E_leak)f mV\n\
gK:\t%(g_K)f μS\n\
EK:\t%(E_K)f mV\n\
C:\t%(C)f nF\n'
        
        # RTDO accounts in mV, nA, μS and is not unit-aware
        params_rtdo = self.params.copy()
        params_rtdo['g_leak'] *= 1e3
        params_rtdo['g_K'] *= 1e3
        for key in ['gK_fast', 'gK_slow']:
            if params_rtdo.has_key(key):
                params_rtdo[key] *= 1e3
                string = string + key + ':\t%(' + key + ')f μS\n'

        return string % params_rtdo
    
    def write(self):
        f = open(self.paramsfile, 'w')
        f.write(self.params_str())
        f.close()