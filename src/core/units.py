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
      Threshold to determine minor change.

    multiplier : float, default: 2
      Threshold to determine sudden change.

    Returns
    -------
    point_flags : 1D numpy array, shape (n_samples,)
      Flag of points, 1 if low peak, 2 if high peak, 0 otherwise.
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
  def refine_peak(self, x, flags, margin = 2, *args, **kwargs):
    """Refining peaks in the signal, by moving flag of peaks in a window.

    Parameters
    ----------
    x : 1D numpy array, shape (n_samples,)
      The input signal.

    flags : 1D numpy array, shape (n_samples,)
      Flag of points, 1 if low peak, 2 if high peak, 0 otherwise.

    margin : int, default: 2
      Number of samples each side of current sample in the data.
      These samples create a window for processing.

    Returns
    -------
    refined_flags : 1D numpy array, shape (n_samples,)
      Flag of points after being refined, 1 if low peak, 2 if high peak, 0 otherwise.
    """

    refined_flags = flags.copy()
    n_samples = flags.shape[0]
    for i in range (margin, n_samples - margin):
      if refined_flags[i] == 1:
        new_peak_id = i - margin + np.argmin(x[i - margin: i + margin + 1])
        refined_flags[i] = 0
        refined_flags[new_peak_id] = 1
      elif refined_flags[i] == 2:
        new_peak_id = i - margin + np.argmax(x[i - margin: i + margin + 1])
        refined_flags[i] = 0
        refined_flags[new_peak_id] = 2
    return refined_flags

  @staticmethod
  def select_boundary(self, gr, flags, tvd, min_thickness = 1, gr_shoulder_threshold = 10):
    """Select peaks in the gr curve to become boundaries.

    Parameters
    ----------
    gr : 1D numpy array, shape (n_samples,)
      The input gamma ray.

    flags : 1D numpy array, shape (n_samples,)
      Flag of points, 1 if low peak, 2 if high peak, 0 otherwise.

    tvd : 1D numpy array, shape (n_samples,)
      The input tv depth.

    min_thickness : float, default: 1
      Minimum of thickness of a unit. Thickness is computed by difference of tvd between the first and the last sample in a unit.

    gr_shoulder_threshold : float, default: 10
      Minium of difference of gr between 2 samples to create a shoulder effect.

    Returns
    -------
    boundary_flags : 1D numpy array, shape (n_samples,)
      Boundary flag of points, 1 if boundary, 0 otherwise.
    """

    n_samples = flags.shape[0]
    boundary_flags = flags.copy()
    left = 0
    for i in range (1, n_samples):
      if boundary_flags[i] != 0:
        if tvd[i] - tvd[left] < min_thickness:
          delta_gr = gr[i] - gr[left]
          if abs(delta_gr) > gr_shoulder_threshold:
            boundary_flags[left] = 0
            boundary_flags[i] = 0
            left = (left + i) // 2
            boundary_flags[left] = 1
          else:
            boundary_flags[i] = 0
        else:
          left = i
    return boundary_flags

  def break_unit(self, *args, **kwargs):
    pass

  def stack_unit(self, *args, **kwargs):
    pass

  def detect_sharp_boundary(self, *args, **kwargs):
    pass

  @staticmethod
  def detect_lithofacies(self, boundary_flags, mud_volume, method = "major", *args, **kwargs):
    """Detecting lithofacy of each units using boundary_flags and mud_volume curve.

    Parameters
    ----------
    boundary_flags : 1D numpy array, shape (n_samples,)
      Boundary flag of points, 1 if boundary, 0 otherwise.

    mud_volume : 1D numpy array, shape (n_samples,)
      The input mud volume curve.

    major : str, options: ['major', 'mean']
      Method to compute lithofacies.

    Returns
    -------
    lithofacies : 1D numpy array, shape (n_samples,)
      Lithofacy of units, sample in the same unit have same lithofacy. Lithofacy is in [1, 2, 3, 4].
    """
    
    idx_set = []
    n_samples = boundary_flags.shape[0]
    lithofacies = np.zeros(n_samples).astype(np.int8)
    
    if method == 'mean':
      for i in range (n_samples):
        idx_set.append(i)
        if boundary_flags[i] != 0 or i == n_samples - 1:
          mud_volume_set = mud_volume[idx_set].copy()
          mean_mud_volume = np.mean(mud_volume_set)
          if mean_mud_volume < 0.1:
            lithofacies[idx_set] = 1
          elif mean_mud_volume < 0.40:
            lithofacies[idx_set] = 2
          elif mean_mud_volume < 0.70:
            lithofacies[idx_set] = 3
          else:
            lithofacies[idx_set] = 4
          idx_set = []
    elif method == 'major':
      litho_set = []
      for i in range (n_samples):
        idx_set.append(i)
        if mud_volume[i] < 0.1:
          litho_set.append(1)
        elif mud_volume[i] < 0.4:
          litho_set.append(2)
        elif mud_volume[i] < 0.7:
          litho_set.append(3)
        else:
          litho_set.append(4)
        if boundary_flags[i] != 0 or i == n_samples - 1:
          lithofacies[idx_set] = np.argmax(np.bincount(litho_set))
          idx_set = []
          litho_set = []  
    else:
      raise ValueError('Method is mean or major')
    
    return lithofacies

  def label_shape_code(self, *args, **kwagrs):
    pass
