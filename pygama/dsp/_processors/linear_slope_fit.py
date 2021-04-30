import numpy as np
from numba import guvectorize

@guvectorize(["void(float32[:], float32[:], float32[:], float32[:], float32[:])",
              "void(float64[:], float64[:], float64[:], float64[:], float64[:])"],
             "(n)->(),(),(),()", nopython=True, cache=True)


def linear_slope_fit(w_in, mean, stdev, slope, intercept):   

    """
    Finds a linear fit, mean and stdev of input wavefunction 
    
    Parameters
    ----------

    w_in : array-like
            Input waveform 
    
    a_mean : float

    stdev : float

    slope : float

    intercept : float
    
    
    """

    mean[0] = stdev[0] = slope[0] = intercept[0] = np.nan

    if (np.isnan(w_in).any()):
        return



    sum_x = sum_x2 = sum_xy = sum_y = mean[0] = stdev[0] = 0
    isum = len(w_in)

    for i in range(len(w_in)):
        sum_x += i 
        sum_x2 += i**2
        sum_xy += (w_in[i] * i)
        sum_y += w_in[i]
        mean += (w_in[i]-mean) / (i+1)
        stdev += (w_in[i]-mean)**2


    stdev /= (isum + 1)
    np.sqrt(stdev, stdev)


    slope[0] = (isum * sum_xy - sum_x * sum_y) / (isum * sum_x2 - sum_x * sum_x)
    intercept[0] = (sum_y - sum_x * slope[0])/isum