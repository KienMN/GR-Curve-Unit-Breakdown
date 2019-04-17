import numpy as np
from .utils import compute_number_of_changing_direction_time

class UnitBreaker(object):
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

  @staticmethod
  def stack_unit(self, gr_smooth, units_boundary,
                min_samples = 15, gr_smooth_threshold = 5,
                *args, **kwargs):
    """Stacking units to create stacking patterns.

    Parameters
    ----------
    gr_smooth : 1D numpy array, shape (n_samples,)
      The input smoothed gamma ray.

    unit_boundary : 1D numpy array, shape (n_samples,)
      Boundary flag of points, 1 if boundary, 0 otherwise.

    min_samples : int, default: 15
      Minimum number of samples in a stack.

    gr_smooth_threshold : float, default: 10
      Threshold of smoothed gamma ray to determine different patterns.

    Returns
    -------
    stacks_label : 1D numpy array, shape (n_samples,)
      Pattern of stacks, value is in [1, 2, 3].
    """

    n_samples = gr_smooth.shape[0]
    stacks_pattern = np.zeros(n_samples).astype(np.int8)
    stacks_boundary = self.detect_changing_direction_point(gr_smooth, epsilon = 0, multiplier = 1000000)

    idx_set = []
    for i in range (n_samples):
      idx_set.append(i)
      if stacks_boundary[i] != 0 or i == n_samples - 1:
        if i - idx_set[0] > min_samples:
          dif = gr_smooth[idx_set[-1]] - gr_smooth[idx_set[0]]
          if dif > gr_smooth_threshold:
            stacks_pattern[idx_set] = 2
          elif dif < -gr_smooth_threshold:
            stacks_pattern[idx_set] = 3
          else:
            stacks_pattern[idx_set] = 1
          idx_set = []
        else:
          stacks_boundary[i] = 0

    stacks_label = np.zeros(n_samples).astype(np.int8)
    idx_set = []

    for i in range (n_samples):
      idx_set.append(i)
      if units_boundary[i] != 0 or i == n_samples - 1:
        stacks_pattern_of_unit = stacks_pattern[idx_set].copy()
        stacks_label[idx_set] = np.argmax(np.bincount(stacks_pattern_of_unit))
        idx_set = []

    return stacks_label

  @staticmethod
  def detect_sharp_boundary(self, gr, boundary_flags, min_diff = 40, *args, **kwargs):
    """Detecting sharp boundaries on gr curve.

    Parameters
    ----------
    gr : 1D numpy array, shape (n_samples,)
      The input gamma ray.

    boundary_flags : 1D numpy array, shape (n_samples,)
      Boundary flag of points, 1 if boundary, 0 otherwise.

    min_diff : float, default: 40
      Minium of difference between 2 gr values to be a sharp boundary.

    Returns
    -------
    sharp_boundary_flags : 1D numpy array, shape (n_samples,)
      Flag of sharp boundaries, 1 if true, 0 otherwise.
    """

    n_samples = gr.shape[0]
    sharp_boundary_flags = np.zeros(n_samples).astype(np.int8)

    for i in range (n_samples):
      if boundary_flags[i] != 0:
        if i < n_samples - 1:
          min_fol = np.min(gr[i + 1: i + 3])
          max_fol = np.max(gr[i + 1: i + 3])
          mean_fol = np.mean(gr[i + 1: i + 3])
          min_prev = np.min(gr[i - 2: i])
          max_prev = np.max(gr[i - 2: i])
          mean_prev = np.mean(gr[i - 2: i])
          if max(abs(min_fol - max_prev), abs(max_fol - min_prev), abs(mean_fol - mean_prev)) > min_diff:
            sharp_boundary_flags[i] = 1
    return sharp_boundary_flags

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
      Lithofacy of units, samples in the same unit have same lithofacy. Lithofacy is in [1, 2, 3, 4].
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

  @staticmethod
  def label_shape_code(self, gr, boundary_flags, tvd, lithofacies, variance,
                      gr_threshold = 8, gr_avg_threshold = 6, tvd_threshold = 2,
                      roc_threshold = 0.2, variance_threshold = 25, change_sign_threshold = 1.5,
                      *args, **kwagrs):
    """Labeling shape of gr curve of each units.

    Parameters
    ----------
    gr : 1D numpy array, shape (n_samples,)
      The input gamma ray.

    boundary_flags : 1D numpy array, shape (n_samples,)
      Boundary flag of points, 1 if boundary, 0 otherwise.

    tvd : 1D numpy array, shape (n_samples,)
      The input tv depth.

    lithofacies : 1D numpy array, shape (n_samples,)
      Lithofacy of units, samples in the same unit have same lithofacy. Lithofacy is in [1, 2, 3, 4].

    variance : 1D numpy array, shape (n_samples,)
      Variance of each units, samples in the same unit have same variance.

    gr_threshold : float, default: 8
      Threshold of difference of gr to detect upward trend.

    gr_avg_threshold : float, default: 6
      Threshold of difference of average gr to detect upward trend.

    tvd_threshold : float, default: 2
      Minimum thickness of unit which can be serrated.
      Thickness is computed by difference of tvd between the first and the last sample in a unit.

    roc_threshold : float, default: 0.2
      Threshold of rate of change

    variance_threshold : float, default: 25
      Threshold of variance

    change_sign_threshold : float, default: 1.5
      Threshold of change sign rate of data.

    Returns
    -------
    labels : 1D numpy array, shape (n_samples,)
      GR shape code of units, samples in the same unit have same lithofacy. Shape code is in [1, 2, 3, 4, 5].
    """

    n_samples = gr.shape[0]
    labels = np.zeros(n_samples).astype(np.int8)
    gr_set = []
    idx_set = []

    for i in range (n_samples):
      idx_set.append(i)
      if boundary_flags[i] != 0 or i == n_samples - 1:
        if lithofacies[i] != 4:
          gr_set = gr[idx_set].copy()
          avg_first = np.average(gr_set[:3])
          max_first = np.amax(gr_set[:3])
          min_first = np.amin(gr_set[:3])
          avg_last = np.average(gr_set[-3:])
          max_last = np.amax(gr_set[-3:])
          min_last = np.amin(gr_set[-3:])
          delta_max_first_min_last = max_first - min_last
          delta_min_first_max_last = min_first - max_last
          delta_avg = avg_first - avg_last
          thickness = tvd[idx_set[-1]] - tvd[idx_set[0]]

          if thickness < tvd_threshold:            
            if delta_max_first_min_last > gr_threshold and abs(delta_avg) > gr_avg_threshold:
              labels[idx_set] = 1
            elif delta_min_first_max_last < -gr_threshold and abs(delta_avg) > gr_avg_threshold:
              labels[idx_set] = 4
            else:
              labels[idx_set] = 2
          else:
              if compute_number_of_changing_direction_time(gr_set) / thickness > change_sign_threshold \
              and variance[idx_set[0]] > variance_threshold \
              and lithofacies[i] != 1:
                labels[idx_set] = 3
              elif delta_max_first_min_last > gr_threshold and abs(delta_avg) > gr_avg_threshold:
                labels[idx_set] = 1
              elif delta_min_first_max_last < -gr_threshold and abs(delta_avg) > gr_avg_threshold:
                labels[idx_set] = 4
              else:
                labels[idx_set] = 2
        else:
          labels[idx_set] = 5
        idx_set = []
        gr_set = []
    return labels