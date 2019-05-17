import pandas as pd
import os

default_filepath = os.path.dirname(__file__) + '/sample_data/sample_dataset.csv'

def load_dataset(filepath = default_filepath):
  """Loading sample dataset or a specific dataset

  Parameters
  ----------
  filepath : str
    Filepath of sample dataset. If specified, file must match pattern of sample dataset.

  Returns
  -------
  gr : 1D numpy array, shape (n_samples,)
    The GR curve
  mud_volume : 1D numpy array, shape (n_samples,)
    The mud volume curve
  tvd : 1D numpy array, shape (n_samples,)
    The TVD curve
  """

  dataset = pd.read_csv(filepath)
  gr = dataset.GR.values
  mud_volume = dataset.MUD_VOLUME.values
  tvd = dataset.TVD.values
  md = dataset.Depth.values
  return gr, mud_volume, tvd, md