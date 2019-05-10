import context

from unit_breakdown.units import UnitBreaker
from unit_breakdown.dataset import load_dataset
from unit_breakdown.smoothing_functions import window_smooth
from unit_breakdown.utils import compute_variance_base_on_slope_line

import numpy as np
import unittest

class TestUnitBreakdown(unittest.TestCase):
  def test_load_dataset(self):
    try:
      gr, _, _ = load_dataset()
    except:
      gr = None
    self.assertIsNotNone(gr)
  
  def test_window_smoothing(self):
    gr, _, _ = load_dataset()
    gr_smooth = window_smooth(gr, window_len = 14, window = 'hamming')
    self.assertAlmostEqual(gr_smooth.shape, gr.shape)
  
  def test_detect_changing_direction_point(self):
    gr, _, _ = load_dataset()
    gr_smooth = window_smooth(gr, window_len = 14, window = 'hamming')
    changing_direction_point_flag = UnitBreaker().detect_changing_direction_point(x = gr_smooth, epsilon = 0.05, multiplier = 7)
    self.assertAlmostEqual(changing_direction_point_flag.shape, gr.shape)

  def test_refine_peak(self):
    gr, _, _ = load_dataset()
    gr_smooth = window_smooth(gr, window_len = 14, window = 'hamming')
    changing_direction_point_flag = UnitBreaker().detect_changing_direction_point(x = gr_smooth, epsilon = 0.05, multiplier = 7)
    refined_peak = UnitBreaker().refine_peak(x = gr, flags = changing_direction_point_flag)
    refined_peak_2 = UnitBreaker().refine_peak(x = gr, flags = refined_peak)
    self.assertEqual(refined_peak_2.shape, gr.shape)

  def test_select_boundary(self):
    gr, _, tvd = load_dataset()
    gr_smooth = window_smooth(gr, window_len = 14, window = 'hamming')
    changing_direction_point_flag = UnitBreaker().detect_changing_direction_point(x = gr_smooth, epsilon = 0.05, multiplier = 7)
    refined_peak = UnitBreaker().refine_peak(x = gr, flags = changing_direction_point_flag)
    refined_peak_2 = UnitBreaker().refine_peak(x = gr, flags = refined_peak)
    boundary_flags = UnitBreaker().select_boundary(gr = gr, flags = refined_peak_2, tvd = tvd,
                                                  min_thickness = 1, gr_shoulder_threshold = 10)
    self.assertEqual(np.max(boundary_flags), 1)

  def test_detect_lithofacies(self):
    gr, v_mud, tvd = load_dataset()
    gr_smooth = window_smooth(gr, window_len = 14, window = 'hamming')
    changing_direction_point_flag = UnitBreaker().detect_changing_direction_point(x = gr_smooth, epsilon = 0.05, multiplier = 7)
    refined_peak = UnitBreaker().refine_peak(x = gr, flags = changing_direction_point_flag)
    refined_peak_2 = UnitBreaker().refine_peak(x = gr, flags = refined_peak)
    boundary_flags = UnitBreaker().select_boundary(gr = gr, flags = refined_peak_2, tvd = tvd,
                                                  min_thickness = 1, gr_shoulder_threshold = 10)
    lithofacies = UnitBreaker().detect_lithofacies(boundary_flags = boundary_flags, mud_volume = v_mud, method = 'major')
    self.assertEqual(lithofacies.shape, gr.shape)

  def test_detect_sharp_boundary(self):
    gr, _, tvd = load_dataset()
    gr_smooth = window_smooth(gr, window_len = 14, window = 'hamming')
    changing_direction_point_flag = UnitBreaker().detect_changing_direction_point(x = gr_smooth, epsilon = 0.05, multiplier = 7)
    refined_peak = UnitBreaker().refine_peak(x = gr, flags = changing_direction_point_flag)
    refined_peak_2 = UnitBreaker().refine_peak(x = gr, flags = refined_peak)
    boundary_flags = UnitBreaker().select_boundary(gr = gr, flags = refined_peak_2, tvd = tvd,
                                                  min_thickness = 1, gr_shoulder_threshold = 10)
    sharp_boundary = UnitBreaker().detect_sharp_boundary(gr = gr, boundary_flags = boundary_flags, min_diff = 40)
    self.assertEqual(sharp_boundary.shape, gr.shape)

  def test_label_shape_code(self):
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
    
    labels = UnitBreaker().label_shape_code(gr = gr, boundary_flags = boundary_flags, tvd = tvd, lithofacies = lithofacies,
                                            variance = variance_2, gr_threshold = 8, gr_avg_threshold = 6, tvd_threshold = 2,
                                            roc_threshold = 0.2, variance_threshold = 40, change_sign_threshold = 2)
    self.assertNotEqual(np.min(labels), 0)

  def test_stack_patterns(self):
    gr, _, tvd = load_dataset()
    
    # Detect unit boundary
    gr_smooth = window_smooth(gr, window_len = 14, window = 'hamming')
    changing_direction_point_flag = UnitBreaker().detect_changing_direction_point(x = gr_smooth, epsilon = 0.05, multiplier = 7)
    refined_peak = UnitBreaker().refine_peak(x = gr, flags = changing_direction_point_flag)
    refined_peak_2 = UnitBreaker().refine_peak(x = gr, flags = refined_peak)
    boundary_flags = UnitBreaker().select_boundary(gr = gr, flags = refined_peak_2, tvd = tvd,
                                                  min_thickness = 1, gr_shoulder_threshold = 10)
    # Detect stack boundary
    gr_smooth = window_smooth(gr, window_len = 50, window = 'hamming')

    stacking_patterns = UnitBreaker().stack_unit(gr_smooth = gr_smooth, units_boundary = boundary_flags,
                                                min_samples = 15, gr_smooth_threshold = 5)
    self.assertNotEqual(np.min(stacking_patterns), 0)

if __name__ == '__main__':
  unittest.main()