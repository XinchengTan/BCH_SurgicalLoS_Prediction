import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from time import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

import globals
from c4_model_perf import MyScorer
import c0_data_prepare as dp
from c1_data_preprocessing import Dataset

DATA_HOME = Path("../Data_new_all")
DATA_DIR = Path("../Data_new_all/ModelInput")
DEPLOY_DATA_DIR = Path(".//Deployment_test_data")
DEPLOY_DEP_FILES_DIR = Path('../Deployment_dep_files')
RESULT_DIR = Path('../Results')


def xgb_cv(Xtrain, ytrain, scorers, refit=globals.SCR_ACC, kfold=5):
  # Tune hyperparameter search
  clf = XGBClassifier(random_state=globals.SEED, use_label_encoder=False)
  param_space = {
    'n_estimators': np.arange(50, 301, 50),
    'max_depth': [3, 4, 5, 6, 7],
    'colsample_bytree': np.arange(0.8, 1.01, 0.1),
    'gamma': [0, 1, 5],
    'min_child_weight': [0.1, 0.3, 1],
  }
  grid_search = GridSearchCV(estimator=clf, param_grid=param_space, scoring=scorers, cv=kfold,
                             refit=refit, return_train_score=True, verbose=0)  # TODO: n_jobs=-1 or other numbers?
  grid_search.fit(Xtrain, ytrain)
  return grid_search


def eval_best_estimator(gs, Xtrain, ytrain, Xtest, ytest):
  # Predict on train & test
  train_pred = gs.predict(Xtrain)
  test_pred = gs.predict(Xtest)
  perf_train = MyScorer.apply_scorers(scorers, ytrain, train_pred)
  print('Training set perf: ')
  for k, v in perf_train.items():
    print(k, v)

  perf_test = MyScorer.apply_scorers(scorers, ytest, test_pred)
  print('\nTest set perf: ')
  for k, v in perf_test.items():
    print(k, v)
  return perf_train, perf_test



if __name__ == '__main__':
  # Prepare dataset
  data_df = dp.prepare_data(DATA_DIR / "historic.csv",
                            DATA_DIR / "cpt_hist.csv",
                            DATA_HOME / "cpt2group.csv",
                            DATA_DIR / "ccsr_hist.csv",
                            DATA_DIR / "medication_hist.csv",
                            exclude2021=False,
                            force_weight=False)
  # Shuffle dataset
  data_df = data_df.sample(frac=1).reset_index(drop=True)

  # Load dataset
  data = Dataset(data_df, outcome=globals.NNT, ftr_cols=globals.FEATURE_COLS_NO_WEIGHT_ALLMEDS,
                 col2decile_ftrs2aggf=globals.DEFAULT_COL2DECILE_FTR2AGGF,
                 test_pct=0.2, discretize_cols=['AGE_AT_PROC_YRS'], scaler='robust')
  print('Loaded historic data')

  # Def scorers
  scorers = MyScorer.get_scorer_dict([
    globals.SCR_ACC, globals.SCR_ACC_BAL, globals.SCR_ACC_ERR1, globals.SCR_ACC_ERR2,
    globals.SCR_OVERPRED, globals.SCR_UNDERPRED])

  # Run CV for model tuning
  gs_xgb = xgb_cv(data.Xtrain, data.ytrain, scorers=scorers, kfold=5)

  # Save result
  pd.DataFrame(gs_xgb.cv_results_).to_csv(RESULT_DIR / 'xgb_cv.csv', index=False)
  joblib.dump(gs_xgb, RESULT_DIR / 'gs_xgb.joblib')

