import numpy as np
import pywt
from statsmodels.robust import mad

def window_smooth(x, window_len=11, window='hanning'):
  """Smoothing the data using a window with requested size.
  
  This method is based on the convolution of a scaled window with the signal.
  The signal is prepared by introducing reflected copies of the signal 
  (with the window size) in both ends so that transient parts are minimized
  in the begining and end part of the output signal.
  
  Parameters
  ----------
  x : 1D numpy array, shape (n_samples,)
    The input signal.

  window_len : int
    The dimension of the smoothing window, should be an odd integer.

  window : str
    The type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    flat window will produce a moving average smoothing.

  Returns
  -------
  y : 1D numpy array, shape (n_samples,)
    The smoothed signal.
  """

  if x.ndim != 1:
    raise ValueError("smooth only accepts 1 dimension arrays.")

  if x.size < window_len:
    raise ValueError("Input vector needs to be bigger than window size.")

  if window_len<3:
    return x

  if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

  s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

  if window == 'flat': #moving average
    w=np.ones(window_len,'d')
  else:
    w=eval('np.'+window+'(window_len)')

  y=np.convolve(w/w.sum(),s,mode='valid')
  return y[(window_len//2-1):-(window_len//2)]

def wavelet_smooth(x, wavelet="db4", level=1):
  """Smoothing the data using discrete wavelet transform

  Parameters
  ----------
  x : 1D numpy array, shape (n_samples,)
    The input signal.

  wavelet : str
    Discrete wavelet transform name.

  level : int
    Level of smooth.

  Returns
  -------
  y : 1D numpy array, shape (n_samples,)
    The smoothed signal.
  """

  # calculate the wavelet coefficients
  coeff = pywt.wavedec(x, wavelet, mode = "per")
  
  # calculate a threshold
  sigma = mad(coeff[-level])
  uthresh = sigma * np.sqrt(2*np.log(len(x)))
  coeff[1:] = (pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:])
  
  # reconstruct the signal using the thresholded coefficients
  y = pywt.waverec(coeff, wavelet, mode = "per")
  return y

def fourier_smooth(x, frequency):
  """Smoothing data using Fourier transform
  
  Parameters
  ----------
  x : 1D numpy array, shape (n_samples,)
    The input signal.

  frequency : int
    Frequency using for filtering data.

  Returns
  -------
  y : 1D numpy array, shape (n_samples,)
    The smoothed signal.
  """

  # transform signal to frequency domain
  rft = np.fft.rfft(x)
  if frequency >= rft.shape[0]:
    raise ValueError("Frequency is not bigger than rft size")
  rft[frequency:] = 0
  
  #reconstruct the signal
  y = np.fft.irfft(rft)
  return y