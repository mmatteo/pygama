import numpy as np
from numba import guvectorize
import math

@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float32, float64[:])"],
             "(n),()->()", nopython=True, cache=True)

def fixed_time_pickoff(w_in, t_in, a_out):
    """
    Fixed time pickoff -- gives the waveform value at a fixed time

    Parameters
    ----------
    w_in : array-like
            Input waveform

    t_in : float
            Time point to find value

    a_out : float
            Output value
    """
    
    a_out[0] = np.nan

    if (np.isnan(w_in).any() or np.isnan(t_in)):
        return

    if (not np.floor(t_in)==t_in):
        raise ValueError('Pickoff Time is not an integer')

    if (not int(t_in) in range(len(w_in))):
        raise ValueError('Time point not in length of waveform')

    a_out[0] = w_in[int(t_in)]
