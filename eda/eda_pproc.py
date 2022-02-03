
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import hvplot
import panel as pn

from .. import globals



def display_pproc_profile(df: pd.DataFrame, freq_k=30, outcome_type=globals.LOS):
  """
  Profile df by visualizing each primary proc LoS distribution

  :param df: a preprocessed dataframe
  :param freq_k: number of most frequent primary procedure groups to visualize
  :return:
  """
  pproc_df = df[['PRIMARY_PROC', outcome_type]].groupby(by=['PRIMARY_PROC'])

  pproc_slider = pn.widgets.DiscreteSlider(name='Primary procedure', options=[], value='')

  return


