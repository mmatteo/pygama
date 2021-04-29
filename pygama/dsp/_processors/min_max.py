import numpy as np
from numba import guvectorize
import math


@guvectorize(["void(float32[:], float32[:], float32[:], float32[:], float32[:])",
              "void(float64[:], float32[:], float32[:], float64[:], float64[:])"],
             "(n)->(),(),(),()", nopython=True, cache=True)


def min_max(w_in, t_min, t_max, a_min, a_max):
    
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

    min_index = 0
    max_index = 0

    for i in range(len(w_in)):
        if w_in[i] < w_in[min_index]:
            min_index = i
        if w_in[i] > w_in[max_index]:
            max_index = i

    a_min[0] = w_in[min_index]
    a_max[0] = w_in[max_index]
    t_min[0] = float(min_index)
    t_max[0] = float(max_index)
