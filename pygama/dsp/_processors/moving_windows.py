import numpy as np
from numba import guvectorize


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)

def moving_window_left(w_in, length, w_out):

    '''
    Applies a moving average window to the waveform from the left

    Parameters
    ----------

    w_in : array-like
            The input waveform

    length : float
              Length of the moving window to be applied

    w_out : array-like
            Output waveform after moving window applied

    '''
    
    w_out[:] = np.nan

    if (np.isnan(w_in).any()):
        return

    if (length < 0 or length>len(w_in)):
        raise ValueError('length is out of range')

    w_out[0]= w_in[0]/length
    for i in range(1, int(length)):
        w_out[i] = w_out[i-1] + w_in[i]/length
    for i in range(int(length), len(w_in)):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-int(length)])/length


@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),()->(n)", nopython=True, cache=True)

def moving_window_right(w_in, length, w_out):

    '''
    Applies a moving average window to the waveform from the right

    Parameters
    ----------

    w_in : array-like
            The input waveform

    length : float
              Length of the moving window to be applied

    w_out : array-like
            Output waveform after moving window applied

    '''

    w_out[:] = np.nan

    if (np.isnan(w_in).any()):
        return

    if (length < 0 or length>len(w_in)) :
        raise ValueError('length is out of range')


    w_out[-1]= w_in[-1] 
    for i in range(len(w_in)-2, len(w_in)-int(length)-1,-1):
        w_out[i] = w_out[i+1] + (w_in[i]-w_out[-1])/length
    for i in range(len(w_in)-int(length)-1, -1, -1):
        w_out[i] = w_out[i+1] + (w_in[i] - w_in[i+int(length)])/length


@guvectorize(["void(float32[:], float32, float32, float32[:])",
              "void(float64[:], float64, float64, float64[:])"],
             "(n),(),()->(n)", nopython=True, cache=True)

def moving_window_multi(w_in, length, num_mw, w_out):

    '''
    Applies a series of moving average window to the waveform alternating between applying form the left and right

    Parameters
    ----------

    w_in : array-like
            The input waveform

    length : float
              Length of the moving window to be applied

    num_mw : float
              Number of moving windows to be applied

    w_out : array-like
            Output waveform after moving window applied

    '''

    w_out[:] = np.nan

    if (np.isnan(w_in).any()):
        return

    if (length < 0 or length>len(w_in)) :
        raise ValueError('length is out of range')
    
    if (num_mw <= 0) :
        raise ValueError('num_mw is out of range')


    wf_buf = w_in.copy()
    for i in range(num_mw):
        
        if i % 2 == 1:
            w_out[-1]= w_in[-1] 
            for i in range(len(w_in)-2, len(w_in)-int(length)-1,-1):
                w_out[i] = w_out[i+1] + (w_in[i]-w_out[-1])/length
            for i in range(len(wf_buf)-(int(length)+1), -1,-1):
                w_out[i] = w_out[i+1] + (wf_buf[i] - wf_buf[i+int(length)])/length
        else:
            w_out[0]= wf_buf[0]/length
            for i in range(1, int(length)):
                w_out[i] = w_out[i-1] + wf_buf[i]/length
            for i in range(int(length), len(w_in)):
                w_out[i] = w_out[i-1] + (wf_buf[i] - wf_buf[i-int(length)])/length
        wf_buf[:] = w_out[:]



@guvectorize(["void(float32[:], float32, float32[:])",
              "void(float64[:], float64, float64[:])"],
             "(n),(),(m)", nopython=True, cache=True)



def avg_current(w_in, length, w_out):
    """
    Calculate the derivative of a WF, averaged across n samples. Dimension of
    deriv should be len(wf) - n

    Parameters
    ----------

    w_in : array-like
            The input waveform

    length : float
              Length of the moving window derivative to be applied

    w_out : array-like
            Output waveform after derivation
    """

    w_out[:] = np.nan

    if (np.isnan(w_in).any()):
        return

    if (length < 0 or length>len(w_in)) :
        raise ValueError('length is out of range')
    

    w_out[:] = w_in[int(length):] - w_in[:-int(length)]
    w_out/=length
