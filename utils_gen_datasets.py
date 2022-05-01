from collections import defaultdict, Counter

from IPython.display import display
from sklearn.utils import shuffle
from time import time
from tqdm import tqdm
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import random

from globals import *
from c1_data_preprocessing import Dataset


# Generate ktrials of randomly selected surg_case_keys for test/validation set
def gen_kfoldCV_test_surg_case_keys(dashb_df: pd.DataFrame, cv=5, save_fp=None) -> Dict:
  assert dashb_df['SURG_CASE_KEY'].is_unique, 'Input dashb_df contains duplicated surg_case_key!'
  assert cv >= 1, 'ktrials must be a positive int!'

  # random shuffle data_df
  dashb_df = dashb_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
  N = dashb_df.shape[0]
  test_size = int(dashb_df.shape[0] / cv)
  kf_test_case_keys = {}
  for k in range(cv):
    test_case_keys = dashb_df['SURG_CASE_KEY'].to_numpy()[k*test_size: min(N, (k+1)*test_size)]
    kf_test_case_keys[k] = test_case_keys

  if save_fp is not None:
    pd.DataFrame(kf_test_case_keys).to_csv(save_fp)
  return kf_test_case_keys


# Generate ktrials of randomly selected surg_case_keys for test/validation set
def gen_ktrials_test_surg_case_keys(dashb_df: pd.DataFrame, ktrials=10, test_pct=0.2, save_fp=None) -> Dict:
  assert dashb_df['SURG_CASE_KEY'].is_unique, 'Input dashb_df contains duplicated surg_case_key!'
  assert ktrials >= 1, 'ktrials must be a positive int!'

  test_size = int(test_pct * dashb_df.shape[0])
  kt_test_case_keys = {}
  for k in range(ktrials):
    test_case_keys = random.sample(dashb_df['SURG_CASE_KEY'].to_list(), test_size)
    kt_test_case_keys[k] = np.array(test_case_keys)

  if save_fp is not None:
    pd.DataFrame(kt_test_case_keys).to_csv(save_fp)
  return kt_test_case_keys


def gen_k_datasets_from_test_case_keys(dashb_df: pd.DataFrame, kt_test_case_keys: Dict, **kwargs) -> Dict:
  kt_datasets = {}
  decileFtr_aggs = kwargs.get('col2decile_ftrs2aggf', DEFAULT_COL2DECILE_FTR2AGGF)
  print('\nDecile feature aggregations:')
  for k, v in decileFtr_aggs.items():
    print(k, v)

  for kt, test_keys in tqdm(kt_test_case_keys.items()):
    dataset_k = Dataset(dashb_df, test_case_keys=test_keys, **kwargs)
    kt_datasets[kt] = dataset_k
  return kt_datasets


# Generate ktrials/kfolds of datasets for each binary outcome cutoff
def gen_k_datasets_from_test_case_keys_binaryClf(dashb_df: pd.DataFrame, kt_test_case_keys: Dict,
                                                 **kwargs) -> Dict[str, Dict]:
  bin_kt_datasets = defaultdict(dict)
  for kt, test_keys in tqdm(kt_test_case_keys.items()):
    for bin_nnt in BINARY_NNT_SET:
      dataset_k = Dataset(dashb_df, test_case_keys=test_keys, outcome=bin_nnt, **kwargs)
      bin_kt_datasets[bin_nnt][kt] = dataset_k

  return bin_kt_datasets


# Validate kfold datasets for each binary outcome share the same test case keys in each fold
def validate_cv_datasets_binaryClf(bin_kt_datasets: Dict):
  kt_bin_datasets = swap_dict_keys(bin_kt_datasets)
  # Validate all test keys in test_raw_df are identical in each fold
  for kf, bin2dataset in kt_bin_datasets.items():
    bin_datasets = list(bin2dataset.values())
    test_keys = bin_datasets[0].test_df_raw['SURG_CASE_KEY']
    for dataset in bin_datasets[1:]:
      equal_test_keys = np.sum(np.in1d(dataset.test_df_raw['SURG_CASE_KEY'].to_list(), test_keys))
      if equal_test_keys != len(test_keys):
        print(f'k = {kf}, outcome = {dataset.outcome} has only {equal_test_keys} / {len(test_keys)} '
              f'identical keys to the first Dataset!')
  return True


def swap_dict_keys(key1_key2_datasets: Dict) -> Dict:
  key2_key1_dataset = defaultdict(dict)
  for k1, key2_datasets in key1_key2_datasets.items():
    for k2, dataset in key2_datasets.items():
      key2_key1_dataset[k2][k1] = dataset
  return key2_key1_dataset
