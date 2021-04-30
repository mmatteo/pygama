import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], float32, int32, int32[:])",
              "void(float64[:], float64, int32, int32[:])",
              "void(int32[:], int32, int32, int32[:])",
              "void(int64[:], int64, int32, int32[:])"],
             "(n),(),()->()", nopython=True, cache=True)

def old_find_tp100(wf_in, energy, tp_0, tp_out):


    for i in range(tp_0, len(wf_in), 1):
        if wf_in[i] >= energy:
            tp_out[0] = i
            return
    tp_out[0] = np.nan
