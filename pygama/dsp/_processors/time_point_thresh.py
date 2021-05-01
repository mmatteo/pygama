import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], float32, float32, float32[:])",
              "void(float64[:], float64, float32, float32[:])"],
             "(n),(),()->()", nopython=True, cache=True)

def time_point_thresh_forward(w_in, a_threshold, t_start, walk_forward, t_out):
    """
    Find the time after/before t_start where w_in first crosses a threshold 

    Parameters
    ----------

     w_in: array-like
           Input waveform
    
     a_threshold: float
                  Threshold to search for

     t_start: float
               Start point to walk forwards from 

     walk_forward: bool
                   Define search direction
    
     tp_out: float
             Final time that waveform is less than threshold
    """
    
    t_out[0] = np.nan

    if (np.isnan(w_in).any() or np.isnan(a_threshold) or np.isnan(t_start) or np.isnan(walk_forward)):
        return

    if (not t_start.is_integer()):
        raise ValueError('Time point is not an integer')

    if (not int(t_start) in range(len(w_in))):
        raise ValueError('Time point not in length of waveform')

    if(walk_forward):
        for i in range(int(t_start), len(w_in)-1):
            if(w_in[i] <= a_threshold < w_in[i+1]):
                t_out[0] = i
                return
    else:
        for i in range(int(t_start), 1, -1):
            if(w_in[i-1] < a_threshold <= w_in[i]):
                t_out[0] = i
                return
