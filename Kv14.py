# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import stfio
from analysis_helpers import *

# Time of one sample in ms
tres = 0.025

# The start and end of the activation step, with some room for the capacitive current to dissipate:
rec_limits = (5050, 44800)
rec_hold = (0, 4925)

# The true step time
rec_step_t0 = 4937

# The start and end of the tail step -- here, the first number is the true step time
rec2_limits = (1087, 40000)
rec2_prepulse = 687

# Index of the first capacitance step
rec3_offset = 322

# Capacitance step spec
rec3_stepdur = 2000
rec3_nsteps = 9

# 1-component kinetics
n1A = [-55., 20.]
h1A = [-48., -5., 0.05]
taun1A = [2., 102., 45., 6., -40.]
tauh1A = [35., 1e4, 72., 7.]

# 2-component kinetics
# # 2018 fit, median voltage
# nsA = [-47.,15.]
# hsA = [-60.,-7.,.02]
# nfA = [-47.,15.]
# hfA = [-60.,-7.,.02]

# taunsA = [4., 62., 85., 28., -20.]
# tauhsA = [120., 10e3, 72., 8.]
# taunfA = [1., 20., 65., 20., -32.]
# tauhfA = [35., 5e3, 72., 7.]

# 2019 classical fit, median voltage
nsA = [-50., 15.]
hsA = [-60.,-7.,.08]
nfA = [-50., 15.]
hfA = [-60.,-7.,.02]

taunsA = [4., 62., 85., 26., -20.]
tauhsA = [120., 5e3, 72., 8.]
taunfA = [1., 22., 65., 18., -32.]
tauhfA = [35., 5e3, 72., 7.]

p_kinetic = np.concatenate((nsA, hsA, nfA, hfA, taunsA, tauhsA, taunfA, tauhfA))
#                           0:2, 2:5, 5:7, 7:10, 10:15, 15:19,  19:24,  24:28

# 2-component time scales, common voltage dependency
# Tentative values
ncA = [-44.2, 17.8]
hcA = [-62., -5., .02]

taunscA = [2., 85., 80., 30., -15.]
tauhscA = [93., 1e4, 80., 5.]
taunfcA = [1., 15., 61., 20., -36.]
tauhfcA = [33., 1e3, 68., 10.]

def sigmoid(p, v):
    return 1/(1+np.exp((p[0]-v)/p[1]))

def sigmoid_min(p, v):
    return p[2] + (1-p[2])/(1+np.exp((p[0]-v)/p[1]))

def taun(p, V):
#   tau_min + tau_max/(np.exp((tau_off+V)/tau_slope1) + np.exp((tau_off+V)/tau_slope2))
    return p[0] + p[1]/(np.exp((p[2]+V)/p[3]) + np.exp((p[2]+V)/p[4]))

def tauh(p, V):
#     $(tau_min) + $(tau_max)/(1.0 + exp(($(tau_mid)-$(V))/$(tau_slope)))
    return p[0] + p[1]/(1. + np.exp((p[2]+V)/p[3]))

def state_at_single(t, V, state, p = np.concatenate((n1A, h1A, taun1A, tauh1A))):
    #                                                0:2, 2:5, 5:10,   10:14
    '''Calculates the 1-component state (n,h) after @a t ms of holding at @a V mV from an initial @a state'''
    
    n1inf = sigmoid(p[0:2], V)
    taun1 = taun(p[5:10], V)
    n = n1inf - (n1inf-state[0]) * np.exp(-t/taun1)

    h1inf = sigmoid_min(p[2:5], V)
    tauh1 = tauh(p[10:14], V)
    h = h1inf - (h1inf-state[1]) * np.exp(-t/tauh1)
    
    return (n, h)

def state_at(t, V, state, p = p_kinetic):
    '''Calculates the 2-component state (ns,hs,nf,hf) after @a t ms of holding at @a V mV from an initial @a state'''
    
    nsinf = sigmoid(p[0:2], V)
    tauns = taun(p[10:15], V)
    ns = nsinf - (nsinf-state[0]) * np.exp(-t/tauns)

    hsinf = sigmoid_min(p[2:5], V)
    tauhs = tauh(p[15:19], V)
    hs = hsinf - (hsinf-state[1]) * np.exp(-t/tauhs)
    
    nfinf = sigmoid(p[5:7], V)
    taunf = taun(p[19:24], V)
    nf = nfinf - (nfinf-state[2]) * np.exp(-t/taunf)

    hfinf = sigmoid_min(p[7:10], V)
    tauhf = tauh(p[24:28], V)
    hf = hfinf - (hfinf-state[3]) * np.exp(-t/tauhf)
    
    return (ns, hs, nf, hf)

def state_at_common(t, V, state, p = np.concatenate((ncA, hcA, taunscA, tauhscA, taunfcA, tauhfcA))):
    #                                               0:2, 2:5, 5:10,  10:14,  14:19,  19:23
    '''
    Calculates the 2-component state (ns,hs,nf,hf) after @a t ms of holding at @a V mV from an initial @a state.
    The kinetics differ from state_at in that n_inf and h_inf are shared between the two components.
    '''
    
    ninf = sigmoid(p[0:2], V)
    hinf = sigmoid_min(p[2:5], V)
    
    tauns = taun(p[5:10], V)
    ns = ninf - (ninf-state[0]) * np.exp(-t/tauns)

    tauhs = tauh(p[10:14], V)
    hs = hinf - (hinf-state[1]) * np.exp(-t/tauhs)
    
    taunf = taun(p[14:19], V)
    nf = ninf - (ninf-state[2]) * np.exp(-t/taunf)

    tauhf = tauh(p[19:23], V)
    hf = hinf - (hinf-state[3]) * np.exp(-t/tauhf)
    
    return (ns, hs, nf, hf)

class Analysis:
    def __init__(self, filebase, in_filenos, out_filenos = (), factor = 1):
        self.savebase = filebase[:-4] % in_filenos[0] + '-' + str(in_filenos[2] or in_filenos[1])
        self.filebase = filebase
        self.out_filenos = out_filenos or (in_filenos[3],)
        self.paramsfile = filebase[:-4] % self.out_filenos[0] + '.params'
        self.params = dict()

        self.rec = read_2channel_ATF(filebase % in_filenos[0], current_factor = factor)
        self.rec2 = read_2channel_ATF(filebase % in_filenos[1], current_factor = factor)
        self.rec3 = read_2channel_ATF(filebase % in_filenos[2]) if in_filenos[2] else None

        self.factor = factor
    
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

        fit_IV(tail_I, tail_V, self.params, 'K')
        
    def fit_gK(self):
        median_voltages = [np.median(V[rec_limits[0]:rec_limits[1]]) for V in self.rec.voltage]
        g_leak = get_gleak(self.rec, self.params['E_leak'], rec_hold)
        peak_currents = [np.max(I[rec_limits[0]:rec_limits[1]]) - self.params['I_leak'](V, g)
                         for I,V,g in zip(self.rec.current, median_voltages, g_leak)]

        self.params['g_A'] = 1.05*peak_currents[-1] / (median_voltages[-1] - self.params['E_K'])
    
    def fit_C(self):
        if self.rec3:
            self.params['C'] = fit_capacitance(self.rec3, tres, rec3_offset, rec3_stepdur, rec3_nsteps)
        else:
            self.params['C'] = fit_capacitance_rec(self.rec, tres, rec_step_t0, rec3_stepdur)

    def params_str(self):
        string = '\
gl:\t%(g_leak)f μS\n\
El:\t%(E_leak)f mV\n\
gA:\t%(g_A)f μS\n\
EK:\t%(E_K)f mV\n\
C:\t%(C)f nF\n'
        
        # RTDO accounts in mV, nA, μS and is not unit-aware
        params_rtdo = self.params.copy()
        params_rtdo['g_leak'] *= 1e3
        params_rtdo['g_A'] *= 1e3
        for key in ['gA_fast', 'gA_slow']:
            if params_rtdo.has_key(key):
                params_rtdo[key] *= 1e3
                string = string + key + ':\t%(' + key + ')f μS\n'

        return string % params_rtdo
    
    def write(self):
        gl = self.params['g_leak']
        for fno in self.out_filenos:
            rec = read_2channel_ATF(self.filebase % fno, current_factor = self.factor)
            buffer_end = len(rec.voltage[0]) / 64
            g = get_gleak(rec, self.params['E_leak'], (0, buffer_end) )
            self.params['g_leak'] = np.mean(g)
            f = open(self.filebase[:-4] % fno + '.params', 'w')
            f.write(self.params_str())
            f.close()
        self.params['g_leak'] = gl