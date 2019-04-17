import numpy as np

class UnitBreaker():
  def __init__(self, gr = None, md = None, tvd = None, mud = None, *args, **kwargs):
    self._gr = gr
    self._md = md
    self._tvd = tvd
    self._mud = mud

  @staticmethod
  def detect_changing_direction_point(self, x, epsilon = 0.02, multiplier = 2, *args, **kwargs):
    """Detecting changing direction points in the signal.

    Parameters
    ----------
    x : 1D numpy array, shape (n_samples,)
      The input signal.

    epsilon : float, default: 0.02
      Threshold to determine minor change

    multiplier : float, default: 2
      Threshold to determine sudden change

    Returns:
    point_flags : 1D numpy array, shape (n_samples,)
      Flag of point, 1 if low point, 2 if high point, 0 otherwise
    """

    diff = np.diff(x, axis = 0)
    n_samples = len(x)
    point_flags = np.zeros(n_samples)
    
    increase = False
    if diff[0] >= 0:
      increase = True
    for i in range (1, n_samples - 1):
      if diff[i] * diff[i - 1] < 0:
        if abs(diff[i]) > epsilon or abs(diff[i - 1]) > epsilon:
          if increase:
            point_flags[i] = 2
          else:
            point_flags[i] = 1
        increase = not increase
      elif diff[i] * diff[i - 1] > 0:
        if abs(diff[i]) > abs(diff[i - 1]) * multiplier:
          if increase:
            point_flags[i] = 1
          else:
            point_flags[i] = 2
        elif abs(diff[i]) < abs(diff[i - 1]) / multiplier:
            if increase:
              point_flags[i] = 2
            else:
              point_flags[i] = 1
        elif diff[i] * diff[i - 1] == 0:
          if increase and diff[i] < 0:
            if abs(diff[i]) > 0:
              point_flags[i] = 2
              increase = not increase
          elif (not increase) and diff[i] > 0:
            if abs(diff[i]) > 0:
              point_flags[i] = 1
              increase = not increase
    return point_flags

  @staticmethod
  def refine_peak(self, x, flag, margin = 2, *args, **kwargs):
    pass

  def break_unit(self, *args, **kwargs):
    pass

  def stack_unit(self, *args, **kwargs):
    pass

  def detect_sharp_boundary(self, *args, **kwargs):
    pass

  def compute_lithofacies(self, *args, **kwargs):
    pass

  def label_shape_code(self, *args, **kwagrs):
    pass
