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
    g = str('g_$').replace('$', postfix)
    E = str('E_$').replace('$', postfix)
    params[g] = fit[0] # mS
    params[E] = -fit[1] / fit[0] # mV
    params[str('I_$').replace('$', postfix)] = lambda V: (V-params[E])*params[g]


# In[5]:

def fit_leak(rec, params, ax = None, step = (5200, 44800), hold = (0, 4925)):
    '''
    Finds the leak conductance parameters from a family of steps (holding to variable potential)
    and deposits the values in params. Argument `step` defines the index range during which
    the voltage is stepped to the test potential, while `hold` defines the pre-step holding duration.
    Only traces stepping at least 4 mV below the holding potential are considered.
    Given the possibility of changing g_leak across the recording, each step is considered separately,
    and the mean g_leak and E_leak found from each hold/step pair saved to params.
    If given an axes to draw on, a full I-V plot is drawn with the data and the best fit line.
    '''

    # Get the peak current and median voltage for each step:
    step_voltages = [np.median(V[step[0]:step[1]]) for V in rec.voltage]
    hold_voltages = [np.median(V[hold[0]:hold[1]]) for V in rec.voltage]
    
    # Get gl/El from each hold/downward-step pair
    sum_g = 0
    sum_E = 0
    n = 0
    for i in range(len(rec.voltage)):
        if hold_voltages[i]-4 > step_voltages[i]:
            Ihold = np.median(rec.current[i][hold[0]:hold[1]])
            Istep = np.median(rec.current[i][step[0]:step[1]])
            gl = (Ihold-Istep) / (hold_voltages[i]-step_voltages[i])
            El = hold_voltages[i] - Ihold/gl
            sum_g += gl
            sum_E += El
            n += 1
    
    # Record the average
    params['g_leak'] = sum_g/n
    params['E_leak'] = sum_E/n
    params['I_leak'] = lambda V, g = params['g_leak']: (V-params['E_leak'])*g

    # Draw
    if ax != None:
        peak_currents = [np.max(I[step[0]:step[1]]) for I in rec.current]
        IVplot(peak_currents, step_voltages, ax)

        leak_plot_V = np.array([-120, 50])
        ax.plot(leak_plot_V, params['I_leak'](leak_plot_V))

def get_gleak(rec, E_leak, hold):
    '''
    Finds the leak conductance for each trace in rec in the window specified by hold (typically the prepulse holding period),
    under the assumption that the specified holding period is uncontaminated by non-leak currents.
    '''
    g = np.zeros(len(rec.voltage))
    for i in range(len(rec.voltage)):
        V = np.median(rec.voltage[i][hold[0]:hold[1]])
        I = np.median(rec.current[i][hold[0]:hold[1]])
        g[i] = I / (V - E_leak)
    return g


# In[6]:


def exp_decay(t, tau, a):
    return a * np.exp(-t / tau)

def exp2_decay(t, p):
    return p[0]*np.exp(-t/p[1]) + p[2]*np.exp(-t/p[3])


# In[7]:

def get_tail_cut(rec2, tail_start):
    '''Returns the index of the latest maximum of the last n current trace's tail current'''
    
    # Locate the deepest point of the (negative) capacitance spike
    tail_min = [np.argmin(I[tail_start:tail_start+500]) + tail_start for I in rec2.current]

    # Locate the highest following point in each trace
    tail_max_med = [(np.argmax(I[tmin:tail_start+500]) + tmin, np.median(I[tmin:tail_start+500]))
                    for I, tmin in zip(rec2.current, tail_min)]
    
    # Disregard traces where the tail current is negative
    tail_max_positives = [t if med > 0 else 0 for t, med in tail_max_med]
    
    latest = max(tail_max_positives)
    if latest == 0:
        latest = tail_max_med[-1][0]
    
    # Return the latest positive peak and the number of negative traces
    return latest, len(tail_max_positives) - np.count_nonzero(tail_max_positives)


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
    rec2.tail_cut, rec2.tail_nnegative = get_tail_cut(rec2, tail_start)
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
                                           args = (t, rec2.tails[i]), method='lm'
                                          )
        tau.append(ret.x[0])
        a.append(ret.x[1])
        
    rec2.tau, rec2.a = tau, a
    return tau, a

def fit_tails_exp2(rec2, tail_start = 4750, tail_end = 44000, median_len = 8000, baseline = None, min_tau = 30):
    '''
    Fits tail currents to a double exponential decay. For more details, see fit_tails.
    Returns: a list of vectors [a1, tau1, a2, tau2] to fit each trace with exp2_decay, which is also deposited in rec2.pdecay
    '''
    rec2.tail_start = tail_start
    rec2.tail_cut, rec2.tail_nnegative = get_tail_cut(rec2, tail_start)
    rec2.tail_voltages = [np.median(rec2.voltage[i][rec2.tail_cut:tail_end]) for i in range(len(rec2.voltage))]
    
    tails = [np.array(I[rec2.tail_cut:tail_end]) for I in rec2.current]
    if type(baseline) == list and len(baseline) == len(tails):
        rec2.tail_baseline = baseline
    else:
        rec2.tail_baseline = [np.median(I[-median_len:]) for I in tails]
    rec2.tails = [tail-base for tail,base in zip(tails,rec2.tail_baseline)]
    
    t = np.arange(len(rec2.tails[0])) + rec2.tail_cut - tail_start
    
    rec2.pdecay = fit_exp2(rec2.tails, t, min_tau)
    return rec2.pdecay
    
    
def fit_exp2(tails, t, min_tau = 1):
    '''Fit tail currents to a double exponential. Both components are forced to have the same sign.
    Returns: a list of vectors [a1, tau1, a2, tau2] to fit each trace with exp2_decay.'''
    # Bounding LM to positive or negative values only using transformation
    # See: http://cars9.uchicago.edu/software/python/lmfit/bounds.html
    # See: https://github.com/jjhelmus/leastsqbound-scipy
    def pos_to_unlimited(p):
        return np.sqrt((p + 1)**2 - 1) # neg_to_unlimited is sqrt((-p+1)**2 - 1)
    def bounded_to_internal(p):
        return (pos_to_unlimited(abs(p[0])), p[1], pos_to_unlimited(abs(p[2])), p[3])
    init_cond = bounded_to_internal((.3,130, .03,7500))
    
    def zero_bounded(p): # Convert unlimited value to bounded positive value
        return -1 + np.sqrt(p**2 + 1)
    def pos_bounded(p):
        return (zero_bounded(p[0]), p[1], zero_bounded(p[2]), p[3])
    def neg_bounded(p):
        return (-zero_bounded(p[0]), p[1], -zero_bounded(p[2]), p[3])

    p = [[]] * len(tails)
    for i in range(len(tails)):
        ret_pos = scipy.optimize.least_squares(lambda p, x, y: (exp2_decay(x, pos_bounded(p)) - y), init_cond,
                                               args = (t, tails[i]), method='lm')
        ret_neg = scipy.optimize.least_squares(lambda p, x, y: (exp2_decay(x, neg_bounded(p)) - y), init_cond,
                                               args = (t, tails[i]), method='lm')
        if i > 0:
            ini = bounded_to_internal(p[i-1])
            ret_ipos = scipy.optimize.least_squares(lambda p, x, y: (exp2_decay(x, pos_bounded(p)) - y), ini,
                                                   args = (t, tails[i]), method='lm')
            ret_ineg = scipy.optimize.least_squares(lambda p, x, y: (exp2_decay(x, neg_bounded(p)) - y), ini,
                                                   args = (t, tails[i]), method='lm')
            rets = (ret_pos, ret_ipos, ret_neg, ret_ineg)
            idx = np.argmin(map(lambda r: r.cost, rets))
            if idx < 2:
                pp = list(pos_bounded(rets[idx].x))
            else:
                pp = list(neg_bounded(rets[idx].x))
        else:
            if ret_pos.cost < ret_neg.cost:
                pp = list(pos_bounded(ret_pos.x))
            else:
                pp = list(neg_bounded(ret_neg.x))
        
        if pp[1] < min_tau: print "Trace", i, "truncated fast tau", pp[1]; pp[0] = 0. 
        if pp[3] < min_tau: print "Trace", i, "truncated fast tau", pp[3]; pp[2] = 0.
        p[i] = tuple(pp)
        
    return p
    
    
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
        plt.plot(tx, rec2.current[i][begin:end] - rec2.tail_baseline[i], alpha=0.5)
    
    plt.gca().set_prop_cycle(None)
    for i in traces:
        plt.plot(tx, exp_decay(t, rec2.tau[i], rec2.a[i]), linewidth = 1)
    
    plt.xlabel('Time after step [ms]')
    plt.ylabel(u'Current [μA]')
    
def plot_tail_fit_exp2(rec2, params, begin, end, tres, traces):
    '''Plots measured and double exponential fitted tail currents.'''
    t = np.arange(begin, end) - rec2.tail_start
    tx = t * tres
    
    plt.gca().set_prop_cycle(None)
    for i in traces:
        plt.plot(tx, rec2.current[i][begin:end] - rec2.tail_baseline[i], alpha=0.5)
    
    plt.gca().set_prop_cycle(None)
    for i in traces:
        plt.plot(tx, exp2_decay(t, rec2.pdecay[i]), linewidth = 1)
    
    plt.xlabel('Time after step [ms]')
    plt.ylabel(u'Current [μA]')
    
    if begin < rec2.tail_cut and end > rec2.tail_cut:
        fit_at_cut = exp2_decay(rec2.tail_cut-rec2.tail_start, np.transpose(rec2.pdecay))
        cut_limits = min(fit_at_cut), max(fit_at_cut)
        plt.ylim(2*cut_limits[0]-cut_limits[1], 2*cut_limits[1]-cut_limits[0])
    
    
def fit_capacitance(rec3, dt, offset, stepdur, nsteps = 9):
    '''
    Calculate and return the capacitance in nF from a series of VC steps between two voltages:
    C = integral(I_step, dt) / dV
    Baseline for the current at each step is the mean current measured during the final 25% of the step.
    '''
    currents = [np.array(rec3.current[0][i:i+stepdur]) for i in np.arange(nsteps)*stepdur + offset]
    voltages = [np.median(rec3.voltage[0][:offset])] + \
               [np.median(rec3.voltage[0][i:i+stepdur]) for i in np.arange(nsteps)*stepdur + offset]
    C = [None]*len(currents)
    
    for (i, I, dV) in zip(range(len(C)), currents, np.diff(voltages)):
        # Find steady-state current
        steadystate = np.median(I[stepdur*3/4:])
        
        # Integrate, correcting for steady state (integral over the baseline section alone ought to be zero)
        C[i] = np.sum(I-steadystate) * dt / dV

    return np.mean(C) * 1e3 # Capacitance in nF


def fit_capacitance_rec(rec, dt, offset, stepdur, nTraces = 7):
    currents = [np.array(I[offset:offset + stepdur])
                for I in rec.current[:nTraces]]
    voltages = [np.median(V[:offset]) - np.median(V[offset:offset + stepdur])
                for V in rec.voltage[:nTraces]]
    spikes = [(I, dV) for I, dV in zip(currents, voltages) if abs(dV) > 5]
    C = [None]*len(spikes)

    for (i, (I, dV)) in zip(range(len(C)), spikes):
        # Find steady-state current
        steadystate = np.median(I[stepdur*3/4:])

        # Integrate, correcting for steady state (integral over the baseline section alone ought to be zero)
        C[i] = -np.sum(I-steadystate) * dt / dV

    return np.mean(C) * 1e3 # Capacitance in nF



def mad(x):
    return np.median(np.absolute(np.median(x)-x))


def linear_exclude_outliers(X, Y, tolerance = 5):
    p = np.polyfit(X, Y, 1)
    residuals = np.array([p[0]*x + p[1] - y for x,y in zip(X, Y)])
    zMAD = (residuals-np.median(residuals))/mad(residuals)
    return np.nonzero(np.absolute(zMAD) < tolerance)

def bounded_LM_fit(fun, p0, args, bounds):
    # see http://cars9.uchicago.edu/software/python/lmfit/bounds.html
    def bounds_funcs(lower, upper):
        internal = lambda x: np.arcsin(2*(x-lower)/(upper-lower) - 1)
        bounded = lambda x: lower + (np.sin(x)+1) * (upper-lower)/2
        return internal, bounded
    def upper_bounds_funcs(upper):
        internal = lambda x: np.sqrt((upper-x+1)**2 - 1)
        bounded = lambda x: upper + 1 - np.sqrt(x*x + 1)
        return internal, bounded
    def lower_bounds_funcs(lower):
        internal = lambda x: np.sqrt((x-lower+1)**2 - 1)
        bounded = lambda x: lower - 1 + np.sqrt(x*x + 1)
        return internal, bounded

    assert len(p0) == len(bounds[0]) == len(bounds[1])

    conv = [None]*len(p0)
    for i, (lo, hi) in enumerate(zip(bounds[0], bounds[1])):
        if lo == -np.inf and hi == np.inf:
            conv[i] = (lambda p:p, lambda q:q)
        elif lo == -np.inf:
            conv[i] = upper_bounds_funcs(hi)
        elif hi == np.inf:
            conv[i] = lower_bounds_funcs(lo)
        else:
            conv[i] = bounds_funcs(lo, hi)

    p0u = [c[0](p) for c,p in zip(conv, p0)]
    funwrap = lambda q, x, y: fun([c[1](p) for c,p in zip(conv, q)], x, y)

    ret = scipy.optimize.least_squares(funwrap, p0u, args=args, method='lm')
    ret.x = np.array([c[1](p) for c,p in zip(conv, ret.x)])
    return ret
