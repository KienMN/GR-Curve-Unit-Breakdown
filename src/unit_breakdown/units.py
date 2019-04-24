import numpy as np
from .utils import compute_number_of_changing_direction_time

class UnitBreaker(object):
  @staticmethod
  def detect_changing_direction_point(x, epsilon = 0.02, multiplier = 2, *args, **kwargs):
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
  def refine_peak(x, flags, margin = 2, *args, **kwargs):
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
  def select_boundary(gr, flags, tvd, min_thickness = 1, gr_shoulder_threshold = 10):
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
    boundary_flags[boundary_flags != 0] = 1
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
    boundary_flags[-1] = 1
    return boundary_flags

  @staticmethod
  def stack_unit(gr_smooth, units_boundary,
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
    stacks_boundary = UnitBreaker().detect_changing_direction_point(gr_smooth, epsilon = 0, multiplier = 1000000)

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
  def detect_sharp_boundary(gr, boundary_flags, min_diff = 40, *args, **kwargs):
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
  def detect_lithofacies(boundary_flags, mud_volume, method = "major", *args, **kwargs):
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
  def label_shape_code(gr, boundary_flags, tvd, lithofacies, variance,
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
      Threshold of rate of change. ROC is computed by number of time the curve cross a line
      (connecting first points and last points) divided by number of samples in a unit.

    variance_threshold : float, default: 25
      Threshold of variance.

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

  @staticmethod
  def assign_unit_index(boundary_flags):
    """Assigning index for units of curve.

    Parameters
    ----------
    boundary_flags : 1D numpy array, shape (n_samples,)
      Boundary flag of points, 1 if boundary, 0 otherwise.

    Returns
    -------
    unit_index : 1D numpy array, shape (n_samples,)
      Index of units, samples in the same unit have same index.
    """

    n_samples = boundary_flags.shape[0]
    unit_index = np.zeros(n_samples).astype(np.int64)
    sequence_number = 0
    idx_set = []
    for i in range (n_samples):
      idx_set.append(i)
      if boundary_flags[i] != 0 or i == n_samples - 1:
        unit_index[idx_set] = sequence_number
        sequence_number += 1
        idx_set = []
    return unit_index

  @staticmethod
  def find_similar_units(gr, tvd, boundary_flags, lithofacies, gr_shape_code, thickness,
                        zcr, slope, mean_unit, variance_1, variance_2,
                        max_depth, min_depth = 0, unit_index = None, return_unit_index = True,
                        rms_threshold = 6, zcr_threshold = 0.5, slope_threshold = 0.5,
                        mean_threshold = 15, variance_1_threshold = 10, variance_2_threshold = 10,
                        weights = {'zcr': 1, 'slope': 5, 'mean': 1, 'variance1': 1, 'variance2': 2, 'rms': 3},
                        score_threshold = 10,
                        *args, **kwargs):
    """Finding similar units on the curve.

    Parameters
    ----------
    gr : 1D numpy array, shape (n_samples,)
      The input gamma ray.

    tvd : 1D numpy array, shape (n_samples,)
      The input tv depth.

    boundary_flags : 1D numpy array, shape (n_samples,)
      Boundary flag of points, 1 if boundary, 0 otherwise.

    lithofacies : 1D numpy array, shape (n_samples,)
      Lithofacy of units, samples in the same unit have same lithofacy. Lithofacy is in [1, 2, 3, 4].

    gr_shape_code : 1D numpy array, shape (n_samples,)
      GR shape code of units, samples in the same unit have same lithofacy. Shape code is in [1, 2, 3, 4, 5].

    thickness : 1D numpy array, shape (n_samples,)
      Thickness of units. Thickness is computed by difference of tvd between the first and the last sample in a unit.

    zcr : 1D numpy array, shape (n_samples,)
      Zero crossing rate, number of time the curve cross a line (connecting first points and last points)
      divided by number of samples in a unit.

    slope : 1D numpy array, shape (n_samples,)
      Slope of units, slope is computed by difference of average of first points and last points divided by number of samples in a unit.

    mean_unit : 1D numpy array, shape (n_samples,)
      Mean value of gr of each units.

    variance_1 : 1D numpy array, shape (n_samples,)
      Variance of each units, corresponding to the line of the mean value.
      Samples in the same unit have same variance.

    variance_2 : 1D numpy array, shape (n_samples,)
      Variance of each units, corresponding to the line connecting first points and last points.

    max_depth : float
      Maximum depth between current unit and comparison unit.

    min_depth : float, default: 0
      Minimum depth between current unit and comparison unit.

    unit_index : 1D numpy array, shape (n_samples,), default: None
      Index of units. Samples in same unit have the same unit index.
    
    return_unit_index : boolean, defaul: True
      If True, returns index of units.

    rms_threshold : float, defaul: 6
      Threshold of root mean square of resampled current unit and resampled comparison unit.

    zcr_threshold : float, default: 0.5
      Threshold of zero crossing rate.

    slope_threshold : float, default: 0.5
      Threshold of slope.

    mean_threshold : float, default: 15
      Threshold of mean value of gr.
    
    variance_1_threshold : float, default: 10
      Threshold of variance 1.
    
    variance_2_threshold : float, default: 10
      Threshold of variance 2.

    weights : dict, default: {'zcr': 1, 'slope': 5, 'mean': 1, 'variance1': 1, 'variance2': 2, 'rms': 3},
      Weights of each criterions.

    score_threshold: int, default: 10
      Threshold of score, if total weights of met criterion is larger than score then 2 units are similar.

    Returns
    -------
    unit_index : 1D numpy array, shape (n_samples,)
      Index of units, if return_unit_index is True.
    
    number_of_similar_pattern : 1D numpy array, shape (n_samples,)
      Number of similar units of each units.

    similar_unit_list : list
      List of list of similar units of each units.
    """

    n_samples = gr.shape[0]
    idx_set = []
    number_of_similar_pattern = np.zeros(n_samples)
    similar_unit_list = []

    if unit_index is None:
      unit_index = UnitBreaker().assign_unit_index(boundary_flags)

    for i in range (n_samples):
      idx_set.append(i)
      if boundary_flags[i] != 0 or i == n_samples - 1:
        sub_idx_set = []
        similar_unit_index = []
        for j in range (i + 1, n_samples):
          sub_idx_set.append(j)
          if boundary_flags[j] != 0 or j == n_samples - 1:
            current_unit_id = idx_set[0]
            comparison_unit_id = sub_idx_set[0]
            if abs(tvd[comparison_unit_id] - tvd[current_unit_id]) >= min_depth and abs(tvd[comparison_unit_id] - tvd[current_unit_id]) <= max_depth:
              if lithofacies[current_unit_id] == lithofacies[comparison_unit_id] and gr_shape_code[current_unit_id] == gr_shape_code[comparison_unit_id]:
                if thickness[current_unit_id] * 0.5 < thickness[comparison_unit_id] and thickness[comparison_unit_id] < thickness[current_unit_id] * 1.5:
                  score = 0
                  n_current_unit_samples = len(idx_set)
                  n_comparision_unit_samples = len(sub_idx_set)
                  current_unit_resamples = np.array([
                    gr[idx_set[0]],
                    gr[idx_set[int(n_current_unit_samples * 0.25)]],
                    gr[idx_set[int(n_current_unit_samples * 0.5)]],
                    gr[idx_set[int(n_current_unit_samples * 0.75)]],
                    gr[idx_set[-1]]
                  ])
                  
                  comparison_unit_resamples = np.array([
                    gr[sub_idx_set[0]],
                    gr[sub_idx_set[int(n_comparision_unit_samples * 0.25)]],
                    gr[sub_idx_set[int(n_comparision_unit_samples * 0.5)]],
                    gr[sub_idx_set[int(n_comparision_unit_samples * 0.75)]],
                    gr[sub_idx_set[-1]]
                  ])
                  
                  rms = np.sqrt(np.mean((current_unit_resamples - comparison_unit_resamples) ** 2))
                  if rms < rms_threshold:
                    score += weights['rms']
                  if abs(zcr[current_unit_id] - zcr[comparison_unit_id]) < zcr_threshold:
                    score += weights['zcr']
                  if abs(slope[current_unit_id] - slope[comparison_unit_id]) < slope_threshold:
                    score += weights['slope']
                  if abs(mean_unit[current_unit_id] - mean_unit[comparison_unit_id]) < mean_threshold:
                    score += weights['mean']
                  if abs(variance_1[current_unit_id] - variance_1[comparison_unit_id]) < variance_1_threshold:
                    score += weights['variance1']
                  if abs(variance_2[current_unit_id] - variance_2[comparison_unit_id]) < variance_2_threshold:
                    score += weights['variance2']
                  if score > score_threshold:
                    number_of_similar_pattern[idx_set] += 1
                    number_of_similar_pattern[sub_idx_set] += 1
                    similar_unit_index.append(unit_index[comparison_unit_id])
            elif abs(tvd[comparison_unit_id] - tvd[current_unit_id]) > max_depth:
              break
            sub_idx_set = []
        similar_unit_list.append(similar_unit_index)
        idx_set = []

    for i in range (len(similar_unit_list)):
      for j in range (len(similar_unit_list[i])):
        if similar_unit_list[i][j] > i:
          similar_unit_list[similar_unit_list[i][j]].append(i)
    if return_unit_index:
      return unit_index, number_of_similar_pattern, similar_unit_list
    return number_of_similar_pattern, similar_unit_list