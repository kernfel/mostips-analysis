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
rec_hold = (0, 4925)

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

# 2018 kinetics
nK = [-13.7, 14.8]
hK = [5.8, -3.9]
sK = [-19.7, 11.1]

taunK = [5., 110., 36., 18., -17.]
tauhK = [200., 8e3, 5e4, -30., -16., 15.]
tausK = [25., 500., 36., 18., -17.]

p_kinetic = np.concatenate((nK, taunK, hK, tauhK, sK, tausK))
#                          0:2, 2:7, 7:9, 9:15, 15:17, 17:22

def sigmoid(p, v):
    return 1/(1+np.exp((p[0]-v)/p[1]))

def taun(p, V):
    return p[0] + p[1]/(np.exp((p[2]+V)/p[3]) + np.exp((p[2]+V)/p[4]))

def tauh(p, V):
    lexp = np.exp((V-p[3])/p[4])
    rexp = np.exp((V-p[3])/p[5])
    return p[0] + p[1]/(1+lexp) + p[2]/(lexp + rexp)

def state_at(t, V, state, p = p_kinetic):
    '''Calculates the state (n,h,s) after @a t ms of holding at @a V mV from an initial @a state'''
    
    _ninf = sigmoid(p[0:2], V)
    _taun = taun(p[2:7], V)
    n = _ninf - (_ninf-state[0]) * np.exp(-t/_taun)

    _hinf = sigmoid(p[7:9], V)
    _tauh = tauh(p[9:15], V)
    h = _hinf - (_hinf-state[1]) * np.exp(-t/_tauh)
    
    _sinf = sigmoid(p[15:17], V)
    _taus = taun(p[17:22], V)
    s = _sinf - (_sinf-state[2]) * np.exp(-t/_taus)
    
    return (n,h,s)

class Analysis:
    def __init__(self, filebase, in_filenos, out_filenos = (), factor = 1, out_factor = 1):
        self.savebase = filebase[:-4] % in_filenos[0] + '-' + str(in_filenos[2] or in_filenos[1])
        self.filebase = filebase
        self.out_filenos = out_filenos or (in_filenos[3],)
        self.paramsfile = filebase[:-4] % self.out_filenos[0] + '.params'
        self.params = dict()

        self.rec = read_2channel_ATF(filebase % in_filenos[0], current_factor = factor)
        self.rec2 = read_2channel_ATF(filebase % in_filenos[1], current_factor = factor)
        self.rec3 = read_2channel_ATF(filebase % in_filenos[2]) if in_filenos[2] else None

        self.factor = factor
        self.out_factor = out_factor

    def fit(self):
        self.fit_leak()
        self.fit_EK()
        self.fit_gK()
        self.fit_C()
    
    def fit_leak(self):
        fit_leak(self.rec, self.params, None, rec_limits, rec_hold)
    
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
        g_leak = get_gleak(self.rec, self.params['E_leak'], rec_hold)
        peak_currents = [np.max(I[rec_limits[0]:rec_limits[1]]) - self.params['I_leak'](V, g)
                         for I,V,g in zip(self.rec.current, median_voltages, g_leak)]

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
                
        string = string % params_rtdo

        if hasattr(self, 'kparams'):
            string = string + self.k_params_str()
        
        return string
        
    def k_params_str(self):
        # (nK, taunK, hK, tauhK, sK, tausK)
        string = """\
gK_fast_k:\t%f μS
gK_slow_k:\t%f μS
nK_mid:\t%f mV
nK_slope:\t%f
taunK_min:\t%f ms
taunK_max:\t%f ms
taunK_off:\t%f ms
taunK_slope1:\t%f
taunK_slope2:\t%f
hK_mid:\t%f mV
hK_slope:\t%f
tauhK_lmin:\t%f ms
tauhK_rmin:\t%f ms
tauhK_max:\t%f ms
tauhK_mid:\t%f ms
tauhK_lslope:\t%f
tauhK_rslope:\t%f
sK_mid:\t%f mV
sK_slope:\t%f
tausK_min:\t%f ms
tausK_max:\t%f ms
tausK_off:\t%f ms
tausK_slope1:\t%f
tausK_slope2:\t%f
"""
        factors = np.ones(len(self.kparams))
        factors[0] = factors[1] = 1e3
        return string % tuple(factors * self.kparams)
    
    def write(self):
        gl = self.params['g_leak']
        for fno in self.out_filenos:
            rec = read_2channel_ATF(self.filebase % fno, current_factor = self.out_factor)
            buffer_end = len(rec.voltage[0]) / 64
            g = get_gleak(rec, self.params['E_leak'], (0, buffer_end) )
            self.params['g_leak'] = np.mean(g)
            f = open(self.filebase[:-4] % fno + '.params', 'w')
            f.write(self.params_str())
            f.close()
        self.params['g_leak'] = gl