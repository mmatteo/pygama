import numpy as np
from numba import guvectorize
import math


@guvectorize(["void(float32[:], float32[:], float32[:],float32[:], float32[:])",
              "void(float64[:], float32[:], float32[:],float64[:], float64[:] )"],
             "(n)->(),(),(),()", nopython=True, cache=True)


def old_min_max(w_in, t_min, t_max, a_min, a_max):
    '''
    Finds the min, max and their time position for a waveform

    Parameters
    ----------

    w_in : array-like
            input waveform
    
    t_min : float
            Output time when waveform is at minimum
    
    t_max : float
            Output time when waveform is at maximum

    a_min : float
            Output value when waveform is at minimum
    
    a_max : float
            Output value when waveform is at maximum

    '''

    a_min[0] = np.nan
    a_max[0] = np.nan
    t_min[0] = np.nan
    t_max[0] = np.nan

    if (np.isnan(w_in).any()):
        return

    a_min[0] = float(math.inf)
    a_max[0] = float(-math.inf)

    for i in len(w_in):
        if w_in[i] < a_min[0]:
            a_min[0] = w_in[i]
            t_min[0] = i.
        if w_in[i] > wf_max[0]:
            a_max[0] = w_in[i]
            t_max[0] = i.
