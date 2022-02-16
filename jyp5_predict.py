"""
These functions makes predictions of all input trained models on the given dataset
"""
import numpy as np
import pandas as pd
import c1_data_preprocessing as dpp
import jyp2_modeling
import globals


# TODO: Fix this with the updated Dataset object!!!
@DeprecationWarning
def gen_all_model_predictions(dataset: dpp.Dataset, all_models: jyp2_modeling.AllModels, from_train=False, subset_size=30):
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
  md2preds[globals.SPS_PRED] = dataset.sps_preds[subset_idx]
  md2preds['CPT_GROUPS'] = dataset.cpt_groups[subset_idx]

  # Order columns and generate dataframe
  columns = ['SURG_CASE_KEY', 'CPT_GROUPS', 'True LOS', globals.SPS_PRED]
  columns.extend(all_models.mdnames)
  df_all = pd.DataFrame(md2preds, columns=columns).sort_values(by=['True LOS'])

  return df_all
