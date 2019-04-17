import numpy as np

def compute_number_of_changing_direction_time(arr):
  diff = np.diff(arr)
  n_samples = diff.shape[0]
  upward = True
  count = 0
  i = 0
  while diff[i] == 0 and i < n_samples:
    i += 1
  if diff[i] < 0 and i < n_samples:
    upward = False
  for j in range (i + 1, n_samples):
    if (upward and diff[j] < 0) or ((not upward) and diff[j] > 0):
      count += 1
      upward = not upward
  return count