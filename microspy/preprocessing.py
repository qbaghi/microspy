import numpy as np
from scipy import signal
from microspy import regression
from pyfftw.interfaces.numpy_fft import fft
from scipy import interpolate


def low_pass_filter(y, n=5, fc=0.05, fs=10.0):

    b, a = signal.butter(n, fc, 'low', analog=False, fs=fs)
    y = signal.filtfilt(b, a, y)

    return y


def cut_and_filter(y_data, n_start, n, fs=10.0, scale=1.0, filtering=False):
    """
    Select a segment of data begening at index n_start with size n
    Parameters
    ----------
    y_data : 2d array
        data to preprocess, has size N x 2 for x,y
    n_start : int
        index of segment starting time
    n : int
        segment size
    fs : float
        sampling frequency
    scale : float
        re-scaling factor to apply to the output

    Returns
    -------
    y : 1d array
        preprocessed data

    """

    y = y_data[n_start:np.int(n_start + n), 1] / scale

    if filtering:
        b, a = signal.butter(4, 0.05, 'low', analog=False, fs=fs)
        y = signal.filtfilt(b, a, y)

    return y


def detrend(y_data, detrend_method='poly', max_order=1, n_knots=10, psd=None):
    """
    Remove linear or polynomial trend of order max_order
    Parameter
    ----------
    y_data : 1d array
        input data of size n
    max_order : int
        maximum order of the polynomial to fit
    psd : 1d array
        noise PSD at Fourier frequencies (size n)

    Returns
    -------
    y_detrend : 1d array
        output detrended data (size n)

    """

    t_norm = np.arange(0, y_data.shape[0])

    if detrend_method == 'poly':

        mat_linear = np.hstack([np.array([t_norm**k]).T
                                for k in range(0, max_order+1)])
        if psd is not None:
            amp = regression.generalized_least_squares(fft(mat_linear, axis=0),
                                                       fft(y_data),  psd)
        else:
            amp = regression.least_squares(mat_linear, y_data)

        trend = np.real(np.dot(mat_linear, amp))
        y_detrend = y_data - trend

    elif detrend_method == 'spline':
        # Detrending using splines
        n_seg = y_data.shape[0] // n_knots
        t_knots = np.linspace(t_norm[n_seg], t_norm[-n_seg], n_knots)
        spl = interpolate.LSQUnivariateSpline(t_norm, y_data, t_knots, k=3,
                                              ext="const")
        trend = spl(t_norm)
        y_detrend = y_data - trend

    return y_detrend, trend


def pre_process(t, y, fc=0.05, n=5, detrend_method='poly', max_order=1,
                n_knots=10, psd=None):
    """

    Parameters
    ----------
    t : ndarray
        time vector
    y : ndarray
        data
    fc : float
        cut-off frequency
    n
    detrend_method
    max_order
    n_knots
    psd

    Returns
    -------

    """

    y_detrend, trend = detrend(y, detrend_method=detrend_method,
                               max_order=max_order,
                               n_knots=n_knots, psd=psd)

    # Filtering
    fs = 1 / (t[1] - t[0])
    b, a = signal.butter(n, fc, 'low', analog=False, fs=fs)
    y_detrend_filt = signal.filtfilt(b, a, y_detrend)

    return y_detrend_filt, y_detrend, trend


def set_to_zero(msk, i, margin):

    msk[np.max([i - margin, 0]):np.min([i + margin, msk.shape[0]])] = 0


def sigma_clip(x, k=5, n_it=3, margin=None, mask_init=None):
    """Short summary.

    Parameters
    ----------
    x : ndarray
        input data
    k : int
        number of folds of standard deviation above which to clip
    n_it : int
        Number of iterations
    margin : int
        margin to take before and after each detection
    mask_init : ndarray
        Initial mask

    Returns
    -------
    ndarray
        Sigma-clipping mask

    """

    if mask_init is None:
        mask = np.ones(x.shape[0])
    else:
        mask = mask_init[:]

    for i in range(n_it):
        inds = np.where(mask == 1)[0]
        m = np.mean(x[inds])
        std = np.std(x[inds])
        mask[np.abs(x - m) > k * std] = 0

    mask2 = mask[:]

    if margin is not None:

        [set_to_zero(mask2, n, margin) for n in np.where(mask == 0)[0]]

    return mask2
