"""
These functions makes predictions of all input trained models on the given dataset
"""

import numpy as np
import pandas as pd
from . import data_preprocessing as dpp
from . import modeling
from . import globals

# [done] 1. set up predict all pipeline --> data table [done]
# [done] 2. Check CPT group and see if there's any bug (cpt_group.nunique, cpt_df,nunique) [done]
#         CPT Group: 893 unique groups
#         CPT dataset: 400 unique CPT groups
#         Dashboard dataset: 396
#         Dashboard dataset (w/o 2021): 395
#         SPS dataset (w/o 2021): 260
# [done] 3. Incorporate surgeon's estimation in all models
# TODO 1. Check primary procedure grouping count, and use them as binary indicator -- plot acc, prec, sensitivity bar plot (descending order)
# TODO 2. Ordinal regression
# TODO 3. Reformat prediction data table
# TODO 4. Calibration


def predict_all(dataset: dpp.Dataset, all_models: modeling.AllModels, from_train=True, subset_size=30):
  """
  Makes prediction for a subset of cases from using all models.

  :param df:
  :param model_map:
  :return: A dataframe of
  """
  data_mat, data_idx, true_y = (dataset.Xtrain, dataset.train_idx, dataset.ytrain) if from_train \
    else (dataset.Xtest, dataset.test_idx, dataset.ytest)
  subset_data_idx = np.random.choice(np.arange(len(data_idx)), subset_size, replace=False)
  subset_idx = data_idx[subset_data_idx]
  Xsubset = data_mat[subset_data_idx, :]

  md2preds = all_models.predict_all(Xsubset)
  md2preds['True LOS'] = true_y[subset_data_idx]
  md2preds['SURG_CASE_KEY'] = dataset.case_keys[subset_idx]
  md2preds[globals.SPS_LOS_FTR] = dataset.sps_preds[subset_idx]
  columns = ['SURG_CASE_KEY', 'True LOS', globals.SPS_LOS_FTR]
  columns.extend(all_models.mdnames)
  df_all = pd.DataFrame(md2preds, columns=columns)

  return df_all
