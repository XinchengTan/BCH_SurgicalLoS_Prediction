import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from time import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


import c0_data_prepare as dp, c1_data_preprocessing as dpp
import utils
import c2_modeling as modeling, jyp4_model_eval as model_eval
from c3_ensemble import Ensemble
from c4_model_perf import MyScorer
import c5_feature_selection as ftr_select
from c7_feature_engineering import FeatureEngineeringModifier, DecileGenerator
from c8_models import get_model
from globals import *


DATA_HOME = Path("../Data_new_all")
DATA_DIR = Path("../Data_new_all/ModelInput")
DEPLOY_DATA_DIR = Path("../Deployment_test_data")
DEPLOY_DEP_FILES_DIR = Path('../Deployment_dep_files')
FS_RESULTS_DIR = Path('../FS_results')


if __name__ == '__main__':
  # Prepare dataset
  data_df = dp.prepare_data(DATA_DIR / "historic3.csv",
                            DATA_DIR / "cpt_hist.csv",
                            DATA_HOME / "cpt2group.csv",
                            DATA_DIR / "ccsr_hist.csv",
                            DATA_DIR / "medication_hist.csv",
                            exclude2021=False,
                            force_weight=False)
  print('Prepared dataset shape: ', data_df.shape)

  # Define Feature selection Experiment Details
  ktrials = 5
  model_abbrs = [KNN, RMFCLF, XGBCLF]
  # No scaling
  print('No scaling')
  ktrial_datasets1 = utils.make_k_all_feature_datasets(data_df, k=ktrials,
                                                       onehot_cols=[PRIMARY_PROC, CPTS, CCSRS],
                                                       discretize_cols=None, scaler=None)
  for model in model_abbrs:
    print('Model %s' % model)
    clf = get_model(model)
    train_perf_df, test_perf_df = ftr_select.FeatureSelector.stepwise_batch_addition_fs(clf, ktrial_datasets1)
    train_perf_df.to_csv(FS_RESULTS_DIR / f'{model}_fs{ktrials}_train_discNA_scalerNA.csv', index=False)
    test_perf_df.to_csv(FS_RESULTS_DIR / f'{model}_fs{ktrials}_test_discNA_scalerNA.csv', index=False)

  # # Robust scaling
  # print('Robust scaling')
  # ktrial_datasets2 = utils.make_k_all_feature_datasets(data_df, k=ktrials, onehot_cols=[PRIMARY_PROC, CPTS, CCSRS],
  #                                                      discretize_cols=None, scaler='robust', scale_numeric_only=True)
  # for model in model_abbrs:
  #   print('Model %s' % model)
  #   clf = get_model(model)
  #   train_perf_df, test_perf_df = ftr_select.FeatureSelector.stepwise_batch_addition_fs(clf, ktrial_datasets2)
  #   train_perf_df.to_csv(FS_RESULTS_DIR / f'{model}_fs{ktrials}_train_discNA_scalerRob.csv', index=False)
  #   test_perf_df.to_csv(FS_RESULTS_DIR / f'{model}_fs{ktrials}_test_discNA_scalerRob.csv', index=False)

  # # Run Sequential Ftr selection w/ models
  # n_feature_vals = [5, 10, 15, 20, 25, 30]
  # model_abbrs = [KNN, RMFCLF, XGBCLF]
  # for model in model_abbrs:
  #   print('Model: %s' % model)
  #   clf = get_model(model)
  #   lgr_results = ftr_select.feature_selection_with_eval(clf, ktrial_datasets1, scorers=None, fs_scorer='accuracy',
  #                                                        n_feature_values=n_feature_vals, fs_how='sfs')
  #
  #
