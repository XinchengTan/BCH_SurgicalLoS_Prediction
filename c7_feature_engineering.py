import numpy as np
import pandas as pd

from . import globals


class FeatureEngineeringModifier(object):

  def __init__(self, df, feature_cols):

    pass


  # Generate pproc sexile & other summary statistics
  def gen_pproc_decile(self, data_df, save_fp=None):
    pproc_median = data_df[['PRIMARY_PROC', globals.LOS]].groupby(by=['PRIMARY_PROC'])[globals.LOS].median()\
      .reset_index(name='PPROC_MedianLOS')
    pproc_mean = data_df[['PRIMARY_PROC', globals.LOS]].groupby(by=['PRIMARY_PROC'])[globals.LOS].mean()\
      .reset_index(name='PPROC_MeanLOS')
    pproc_count = data_df[['PRIMARY_PROC', globals.LOS]].groupby(by=['PRIMARY_PROC']).size()\
      .reset_index(name='PPROC_COUNT')
    pproc_std = data_df[['PRIMARY_PROC', globals.LOS]].groupby(by=['PRIMARY_PROC'])[globals.LOS].std()\
      .reset_index(name='PPROC_SD')
    pproc_min = data_df[['PRIMARY_PROC', globals.LOS]].groupby(by=['PRIMARY_PROC'])[globals.LOS].min()\
      .reset_index(name='PPROC_MinLOS')
    pproc_max = data_df[['PRIMARY_PROC', globals.LOS]].groupby(by=['PRIMARY_PROC'])[globals.LOS].max()\
      .reset_index(name='PPROC_MaxLOS')

    pproc_decile = pproc_median\
      .join(pproc_mean.set_index('PRIMARY_PROC'), on='PRIMARY_PROC', how='left')\
      .join(pproc_count.set_index('PRIMARY_PROC'), on='PRIMARY_PROC', how='left')\
      .join(pproc_std.set_index('PRIMARY_PROC'), on='PRIMARY_PROC', how='left')\
      .join(pproc_min.set_index('PRIMARY_PROC'), on='PRIMARY_PROC', how='left')\
      .join(pproc_max.set_index('PRIMARY_PROC'), on='PRIMARY_PROC', how='left')

    pproc_decile['PPROC_DECILE'] = pproc_decile['PPROC_MedianLOS'].apply(lambda x: float(min(round(x), globals.MAX_NNT + 1)))
    pproc_decile.head()
    pproc_decile.to_csv(save_fp, index=False)
    return pproc_decile

  def gen_medication_decile(self, data_df, save_fp=None):

    return

  # TODO: move all discretization and one-hot encoding here! -- do not one-hot encode medication in data_prepare
  # TODO: gen temporal features


