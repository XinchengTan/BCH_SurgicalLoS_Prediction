import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from time import time


import c0_data_prepare as dp, c1_data_preprocessing as dpp
import utils
from c3_ensemble import Ensemble
from c4_model_perf import MyScorer
import c5_feature_selection as ftr_select
from c7_feature_engineering import FeatureEngineeringModifier, DecileGenerator
from c2_models import get_model
from globals import *


DATA_HOME = Path("../Data_new_all")
DATA_DIR = Path("../Data_new_all/ModelInput")
DEPLOY_DATA_DIR = Path("../Deployment_test_data")
DEPLOY_DEP_FILES_DIR = Path('../Deployment_dep_files')
FS_RESULTS_DIR = Path('../FS_results/stepwise')
FS_RESULTS_TRAIN_DIR = Path('../FS_results/stepwise/train')
FS_RESULTS_TEST_DIR = Path('../FS_results/stepwise/test')


if __name__ == '__main__':
  # Parse args
  parser = argparse.ArgumentParser(description='Stepwise Batch Feature Enhancement')
  parser.add_argument('--scaler', default=None, type=str)
  parser.add_argument('--discretize', '--discr', default=None, nargs='+')  # age, miles, weight
  parser.add_argument('--ktrials', '--kt', default=5, type=int)
  args = parser.parse_args()

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
  suffix = f'kt{args.ktrials}'
  disc_cols = []
  if args.discretize is None:
    disc_cols = None
    suffix += '_discrNA'
  else:
    suffix += '_discr'
    for col in disc_cols:
      col = col.lower()
      if col == 'age':
        disc_cols.append(AGE)
        suffix += 'Age'
      elif col == 'miles':
        disc_cols.append(MILES)
        suffix += 'Miles'
      elif col == 'weight':
        disc_cols.append(WEIGHT_ZS)
        suffix += 'Wt'
      else:
        continue
  suffix += '_' + str(args.scaler)
  print('Input Scaler: ', args.scaler)

  # Generate 'ktrials' of datasets
  #model_abbrs = [LGR, KNN, RMFCLF, XGBCLF]
  model_abbrs = [KNN]
  ktrial_datasets = utils.make_k_all_feature_datasets(data_df, k=args.ktrials,
                                                       onehot_cols=[PRIMARY_PROC, CPTS, CCSRS, DRUG_COLS[3]],
                                                       discretize_cols=disc_cols, scaler=args.scaler)
  # For each model, stepwise add features from training data, and evaluate performance
  for model in model_abbrs:
    print('Model %s' % model)
    clf = get_model(model)
    train_perf_df, test_perf_df = ftr_select.FeatureSelector.stepwise_batch_addition_fs(clf, ktrial_datasets)
    train_perf_df.to_csv(FS_RESULTS_TRAIN_DIR / f'{model}_{suffix}.csv', index=False)
    test_perf_df.to_csv(FS_RESULTS_TEST_DIR / f'{model}_{suffix}.csv', index=False)
