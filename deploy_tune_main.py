# Deployment script for training model on historical dataset
import joblib
import numpy as np
import pandas as pd
import time
from copy import deepcopy
from pathlib import Path
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from typing import Dict, Iterable

from c0_data_prepare import prepare_data
from c1_data_preprocessing import Dataset
from c2_models_train_tune import tune_model_gridSearch, tune_model_optuna, tune_model_randomSearch
from c4_model_perf import eval_model_all_ktrials
from globals import *
from globals_fs import *


def init_result_dir(parent_dir, dir_name):
  path = Path(parent_dir) / str(dir_name)
  try:
    path.mkdir(parents=True, exist_ok=False)
  except FileExistsError:
    print("Result folder already exists")
  else:
    print("Result folder was created")
  return path


# Returns a dict for decile feature aggregation function mapping
def get_decileFtr_config(verbose=False) -> Dict:
  # Modify the following to use different aggregation functions for different decile features
  decileFtr_config = deepcopy(DEFAULT_COL2DECILE_FTR2AGGF)
  decileFtr_config[PPROC]['PPROC_COUNT'] = 'sum'
  decileFtr_config[CPT]['CPT_COUNT'] = 'sum'
  decileFtr_config[CCSR]['CCSR_COUNT'] = 'sum'
  decileFtr_config[MED123]['MED123_COUNT'] = 'sum'
  decileFtr_config[CPT]['CPT_DECILE'] = 'sum'

  print('\n[tune_main] Loaded decile feature aggregation config!\n')
  if verbose:
    for k, v in decileFtr_config.items():
      print(k, v)
  return decileFtr_config


# Initialize a dict of {model abbreviation: None} to save trained classifier object
def init_md_to_clf(md_list: Iterable) -> Dict:
  md_to_clf = {md: None for md in md_list}
  return md_to_clf


if __name__ == '__main__':
  outcome = NNT
  force_weight = False
  scorers = [SCR_ACC, SCR_ACC_ERR1, SCR_OVERPRED0, SCR_UNDERPRED0, SCR_RMSE]
  time_id = time.ctime()
  md_list = [XGBCLF]  # LGR, KNN, RMFCLF,
  result_dir = init_result_dir(AGGREGATIVE_RESULTS_DIR, time_id)

  # 1. Generate training set dataframe with all sources of information combined
  hist_data_df = prepare_data(data_fp=DATA_DIR / "historic3.csv",
                              cpt_fp=DATA_DIR / "cpt_hist.csv",
                              cpt_grp_fp=CPT_TO_CPTGROUP_FILE,
                              ccsr_fp=DATA_DIR / "ccsr_hist.csv",
                              medication_fp=DATA_DIR / "medication_hist.csv",
                              chews_fp=DATA_HOME / "chews_raw/chews_hist.csv",
                              exclude2021=False,
                              force_weight=False)
  if force_weight:
    hist_data_df = hist_data_df[hist_data_df[WEIGHT_ZS].notnull()]
  print(f'\n[tune_main] Loaded os_data_df! Shape: {hist_data_df.shape}')

  # 2. Preprocess & Engineer Features on training data -> Dataset() object
  hist_dataset = Dataset(df=hist_data_df, outcome=outcome,
                         ftr_cols=FEATURE_COLS_NO_WEIGHT_ALLMEDS,
                         col2decile_ftrs2aggf=get_decileFtr_config(),
                         onehot_cols=[CCSRS],
                         discretize_cols=['AGE_AT_PROC_YRS'],
                         scaler='robust', scale_numeric_only=True,
                         remove_o2m=(True, True),
                         test_pct=0)
  print(f'\n[tune_main] Finished data preprocessing and feature engineering! '
        f'hist_dataset.Xtrain shape: {hist_dataset.Xtrain.shape}, '
        f'hist_dataset.ytrain shape: {hist_dataset.ytrain.shape}\n')

  # TODO: 2. understand refit
  # 3. Tune models
  for md in tqdm(md_list):
    # 3.1 Tune each classifier
    print(f'[tune_main] Start to tune {md}')
    search = tune_model_randomSearch(md, hist_dataset.Xtrain, hist_dataset.ytrain,
                                     kfold=5,
                                     scorers=scorers,
                                     n_iters=3,
                                     refit=False)
    # 3.2 Save CV results
    pd.DataFrame(search.cv_results_).to_csv(result_dir / f'{md}_CV_results.csv', index=False)

  # 4. Save feature list
  with open(result_dir / f'feature_list.txt', 'wb') as ftrs_file:
    joblib.dump(hist_dataset.feature_names, ftrs_file)
  print(f'[tune_main] Saved feature list!')



  # # 2. Leave 20% data for testing
  # np.random.seed(SEED)
  # test_case_keys = np.random.choice(hist_data_df['SURG_CASE_KEY'].to_list(),
  #                                   size=hist_data_df.shape[0] * 0.2,
  #                                   replace=False)
  # pd.DataFrame({'test_case_key': test_case_keys}).to_csv(DATA_HOME / f'{outcome}-tuning_test_keys.csv', index=False)

  # with open(PRETRAINED_CLFS_DIR / f'{time_id}_{md}clf.joblib', 'wb') as md_file:
  #   joblib.dump(clf, md_file)
  # print(f'[tune_main] Saved "{md}" to "{time_id}_{md}clf.joblib"')

  # 5. Evaluate performance
  # train_perf_resp, test_perf_resp = eval_model_all_ktrials(
  #   {0: hist_dataset},
  #   {0: md_to_clf},
  #   eval_by_cohort=None,
  #   scorers=DEFAULT_SCORERS_BINCLF + [SCR_AUC],
  # )

  # 6. Save performance of the best of each classifier, and the feature list
  # save_fp = Path(AGGREGATIVE_RESULTS_DIR / f'{time_id}_tuned_clf_perf.csv')
  # test_perf_resp.to_csv(save_fp, index=False)
