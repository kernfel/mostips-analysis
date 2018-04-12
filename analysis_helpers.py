# coding: utf-8
# Helper functions for classical voltage clamp analysis

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import stfio


# In[2]:


def read_2channel_ATF(filename, current_factor = 1, voltage_factor = 1):
    rec = stfio.read(filename)
    rec.current = [rec[0][k].asarray() * current_factor for k in range(0, len(rec[0]), 2)]
    rec.voltage = [rec[0][k].asarray() * voltage_factor for k in range(1, len(rec[0]), 2)]
    return rec


# In[3]:


def IVplot(I, V, ax, style = '*-', Ilabel = u'Current [μA]', Vlabel = 'Voltage [mV]'):
    '''Draws a nicely formatted I-V plot in the given axes.'''
    ax.plot(V, I, style)
    ax.set_xlabel(Vlabel)
    ax.set_ylabel(Ilabel)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    Irange, Vrange = max(I)-min(I), max(V)-min(V)
    if min(I) > 0.2*Irange or max(I) < -0.2*Irange:
        ax.spines['bottom'].set_position(('outward', 5))
    else:
        ax.spines['bottom'].set_position('zero')

    if max(V) > -2 * np.mean(np.diff(V)):
        ax.spines['left'].set_position('zero')
    else:
        ax.spines['left'].set_position(('outward', 5))


# In[4]:


def fit_IV(currents, voltages, params, postfix):
    '''
    Fits the conductance and equilibrium potential to the I-V relationship, depositing results as "g_$postfix$",
    "E_$postfix$", and function "I_$postfix$" in the params dict.
    '''
    fit = np.polyfit(voltages, currents, 1)
    params[str('g_$').replace('$', postfix)] = fit[0] # mS
    params[str('E_$').replace('$', postfix)] = -fit[1] / fit[0] # mV
    params[str('I_$').replace('$', postfix)] = lambda V: V*fit[0] + fit[1]


# In[5]:

def fit_leak(rec, params, ax = None, limits = (5200, 44800)):
    '''
    Finds the leak conductance parameters from a family of steps (holding to variable potential)
    and deposits the values in params. Argument `limits` defines the index range during which
    the voltage is stepped to the test potential.
    If given an axes to draw on, an I-V plot is drawn with the data and the best fit line.
    '''

    # Get the peak current and median voltage for each step:
    peak_currents = [np.max(I[limits[0]:limits[1]]) for I in rec.current]
    median_voltages = [np.median(V[limits[0]:limits[1]]) for V in rec.voltage]

    # Get the median currents to fit the leak current
    median_currents = [np.median(I[limits[0]:limits[1]]) for I in rec.current]

    # Select traces with V <= -60 (could count, instead)
    leak_traces = [i for i in range(len(median_voltages)) if median_voltages[i] <= -60]
    leak_currents = [median_currents[i] for i in leak_traces]
    leak_voltages = [median_voltages[i] for i in leak_traces]
    
    # Fit
    fit_IV(leak_currents, leak_voltages, params, 'leak')

    # Draw
    if ax != None:
        IVplot(peak_currents, median_voltages, ax)

        leak_plot_V = np.array([-120, 50])
        ax.plot(leak_plot_V, params['I_leak'](leak_plot_V))


# In[6]:


def exp_decay(t, tau, a):
    return a * np.exp(-t / tau)


# In[7]:

def get_tail_cut(rec2, tail_start):
    '''Returns the index of the maximum of the last current trace's tail current'''
    
    # Locate the deepest point of the (negative) capacitance spike
    tail_min = np.argmin(rec2.current[-1][tail_start:tail_start+1000]) + tail_start
    
    # Find the highest point of the following tail current, using the last (i.e., highest-voltage) trace
    tail_cut = np.argmax(rec2.current[-1][tail_min:tail_start+1000]) + tail_min
    return tail_cut


def fit_tails(rec2, tail_start = 4750, tail_end = 44000, median_len = 4000, baseline = None):
    '''
    Fits tail currents to an exponential decay. The fitting starts at the maximum of the final trace
    of the recording, which allows the capacitance spike to be discounted. t is considered to be 0 at tail_start.
    The median current in [tail_end-median_len, tail_end] or, if provided, the `baseline` argument is subtracted
    from the currents to ensure a decay to zero. If provided, baseline must be a list of values corresponding
    to rec2.current.
    Returns: tau and A to fit each trace's decay as I = A*exp(-t/tau)
    '''
    rec2.tail_start = tail_start
    rec2.tail_cut = get_tail_cut(rec2, tail_start)
    rec2.tail_voltages = [np.median(rec2.voltage[i][rec2.tail_cut:tail_end]) for i in range(len(rec2.voltage))]
    
    tails = [np.array(I[rec2.tail_cut:tail_end]) for I in rec2.current]
    if type(baseline) == list and len(baseline) == len(tails):
        rec2.tail_baseline = baseline
    else:
        rec2.tail_baseline = [np.median(I[-median_len:]) for I in tails]
    rec2.tails = [tails[i] - rec2.tail_baseline[i] for i in range(len(tails))]
    
    t = np.arange(len(rec2.tails[0])) + rec2.tail_cut - tail_start

    tau, a = [], []
    for i in range(len(rec2.tails)):
        ret = scipy.optimize.least_squares(lambda p, x, y: (exp_decay(x, p[0], p[1]) - y),
                                           (tau[-1], a[-1]) if i>0 else (4000,-0.05),
                                           args = (t, rec2.tails[i])
                                          )
        tau.append(ret.x[0])
        a.append(ret.x[1])
        
    rec2.tau, rec2.a = tau, a
    return tau, a


# In[8]:


def plot_tail_fit(rec2, params, begin, end, tres, traces):
    '''
    Plots measured (bold) and fitted (fine) tail currents together in the time interval
    indicated by [begin, end] and for all indicated traces.
    '''
    t = np.arange(begin, end) - rec2.tail_start
    tx = t * tres
    
    plt.gca().set_prop_cycle(None)
    for i in traces:
        plt.plot(tx, rec2.current[i][begin:end] - rec2.tail_baseline[i])
    
    plt.gca().set_prop_cycle(None)
    for i in traces:
        plt.plot(tx, exp_decay(t, rec2.tau[i], rec2.a[i]), linewidth = 1)
    
    plt.xlabel('Time after step [ms]')
    plt.ylabel(u'Current [μA]')

