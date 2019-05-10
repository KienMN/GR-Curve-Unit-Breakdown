from units import UnitBreaker
from dataset import load_dataset
from smoothing_functions import window_smooth
from utils import compute_variance_base_on_slope_line

import numpy as np

gr, v_mud, tvd = load_dataset()

gr_smooth = window_smooth(gr, window_len = 14, window = 'hamming')
changing_direction_point_flag = UnitBreaker().detect_changing_direction_point(x = gr_smooth, epsilon = 0.05, multiplier = 7)

refined_peak = UnitBreaker().refine_peak(x = gr, flags = changing_direction_point_flag)
refined_peak_2 = UnitBreaker().refine_peak(x = gr, flags = refined_peak)
boundary_flags = UnitBreaker().select_boundary(gr = gr, flags = refined_peak_2, tvd = tvd,
                                              min_thickness = 1, gr_shoulder_threshold = 10)

lithofacies = UnitBreaker().detect_lithofacies(boundary_flags = boundary_flags, mud_volume = v_mud, method = 'major')

n_samples = gr.shape[0]
variance_2 = np.zeros(n_samples)

idx_set = []
for i in range (0, n_samples):
  idx_set.append(i)
  if boundary_flags[i] == 1 or i == n_samples - 1:
    gr_set = gr[idx_set].copy()
    variance_2[idx_set] = compute_variance_base_on_slope_line(gr_set)
    idx_set = []

labels = UnitBreaker().label_shape_code(gr = gr, boundary_flags = boundary_flags,
                                        tvd = tvd, lithofacies = lithofacies,
                                        variance = variance_2, gr_threshold = 8,
                                        gr_avg_threshold = 6, tvd_threshold = 2,
                                        roc_threshold = 0.2, variance_threshold = 40,
                                        change_sign_threshold = 2)

UnitBreaker().visualize_units(gr, labels, boundary_flags, start = 0, n_samples = 1000, n_pics = 6)