import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], float32, float32, float32[:])",
              "void(float64[:], float64, float32, float32[:])"],
             "(n),(),()->()", nopython=True, cache=True)
def time_point_thresh_back(w_in, a_threshold, t_start, t_out):
    """
    Find the last timepoint before t_start that w_in crosses a threshold walking backwards through the waveform

    Parameters
    ----------

     w_in: array-like
                Input waveform
    
     a_threshold: float
                Threshold to search for

     t_start : float
                Start point to walk backwards from 
    
     tp_out: float
                Final time that waveform is less than threshold
    """
    t_out[0] = np.nan

    if (np.isnan(w_in).any() or np.isnan(a_threshold) or np.isnan(t_start)):
        return

    if (not  0 <= t_start <= len(w_in)):
        raise ValueError('t_start must be within waveform')
    
    
    for i in range(int(t_start), -1, -1):
        if(w_in[i]>a_threshold and w_in[i-1]<a_threshold):
            t_out[0] = i
            return

@guvectorize(["void(float32[:], float32, float32, float32[:])",
              "void(float64[:], float64, float32, float32[:])"],
             "(n),(),()->()", nopython=True, cache=True)
def time_point_thresh_forward(w_in, a_threshold, t_start, t_out):
    """
    Find the last timepoint after tp_max before w_in crosses a threshold walking forwards through the waveform

    Parameters
    ----------

     w_in: array-like
                Input waveform
    
     a_threshold: float
                Threshold to search for

     t_start : float
                Start point to walk forwards from 
    
     tp_out: float
                Final time that waveform is less than threshold
    """
    
    t_out[0] = np.nan

    if (np.isnan(w_in).any() or np.isnan(a_threshold) or np.isnan(t_start)):
        return

    if (not  0 <= t_start <= len(w_in)):
        raise ValueError('t_start must be within waveform')
    
    
    for i in range(int(t_start), len(w_in)+1, 1):
        if(w_in[i]<a_threshold and w_in[i+1]>a_threshold):
            t_out[0] = i
            return
