  
import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], int32, float32[:])",
              "void(float64[:], int32, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)

def moving_window_left(wf_in, length, wf_out):
    wf_out[0]= wf_in[0]/length
    for i in range(1, length):
        wf_out[i] = wf_out[i-1] + wf_in[i]/float(length)
    for i in range(length, len(wf_in)):
        wf_out[i] = wf_out[i-1] + (wf_in[i] - wf_in[i-length])/float(length)


@guvectorize(["void(float32[:], int32, float32[:])",
              "void(float64[:], int32, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)

def moving_window_right(wf_in, length, wf_out):
    wf_out[-1]= wf_in[-1]/length
    for i in range(len(wf_in)-2, len(wf_in)-length-1,-1):
        wf_out[i] = wf_out[i+1] + wf_in[i]/float(length)
    for i in range(len(wf_in)-length-1, -1, -1):
        wf_out[i] = wf_out[i+1] + (wf_in[i] - wf_in[i+length])/float(length)

@guvectorize(["void(float32[:], int32, int32, float32[:])",
              "void(float64[:], int32, int32, float64[:])"],
             "(n),(),()->(n)", nopython=True, cache=True)

def moving_window_multi(wf_in, length, no, wf_out):
    wf_buf = wf_in.copy()
    for i in range(no):
        
        if i % 2 == 1:
            wf_out[-1]= wf_buf[-1]/length
            for i in range(len(wf_buf)-2, len(wf_buf)-length-1,-1):
                wf_out[i] = wf_out[i+1] + wf_buf[i]/float(length)
            for i in range(len(wf_buf)-(length+1), -1,-1):
                wf_out[i] = wf_out[i+1] + (wf_buf[i] - wf_buf[i+length])/float(length)
        else:
            wf_out[0]= wf_buf[0]/length
            for i in range(1, length):
                wf_out[i] = wf_out[i-1] + wf_buf[i]/float(length)
            for i in range(length, len(wf_in)):
                wf_out[i] = wf_out[i-1] + (wf_buf[i] - wf_buf[i-length])/float(length)
        wf_buf[:] = wf_out[:]
