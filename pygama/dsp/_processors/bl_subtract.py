import numpy as np
from numba import guvectorize


@guvectorize("void(float32[:], uint16, float32[:])",
             "(n),()->(n)", nopython=True, cache=True)



def bl_subtract(wf_in, baseline, wf_out):

    wf_out[:] = wf_in[:] - float(baseline)
