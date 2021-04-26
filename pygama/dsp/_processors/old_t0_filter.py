import numpy as np
from numba import guvectorize

def old_t0_filter_inv(rise,fall):

    t0_kern = np.arange(2/float(rise),0, -2/(float(rise)**2))
    t0_kern = np.append(t0_kern, np.zeros(int(fall))-(1/float(fall)))

    @guvectorize(["void(float32[:], float32[:])",
                  "void(float64[:], float64[:])",
                  "void(int32[:], int32[:])",
                  "void(int64[:], int64[:])"],
                 "(n),(m)", forceobj=True)
    def t0_filter_inv_out(wf_in,wf_out):
        wf_out[:] = np.convolve(wf_in, t0_kern)[:len(wf_in)]
    return t0_filter_inv_out


