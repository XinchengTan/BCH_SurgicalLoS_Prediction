import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, Hashable, List, Iterable

from globals import *
from c1_data_preprocessing import Dataset
from c4_model_perf import MyScorer, append_perf_row_generic, format_perf_df
from c2_models_nnt import get_model_by_cohort, SafeOneClassWrapper
from c9_cohort_modeling_perf import eval_cohort_clf


# helper function to filter dashb_df by cohort count (keep cases whose cohart has >= 'min_cohort_size' count)
def filter_df_by_cohort_count(dashb_df: pd.DataFrame,  cohort_col: str, min_cohort_size: int):
  cohort_groupby = dashb_df.dropna(subset=[cohort_col]).groupby(cohort_col)
  if min_cohort_size > 1:
    filtered_cohort = cohort_groupby.filter(lambda x: len(x) >= min_cohort_size)
    return filtered_cohort
  return pd.DataFrame(dashb_df)


def gen_ktrial_cohort_test_case_keys(dashb_df: pd.DataFrame, cohort_type: str, min_cohort_size=50,
                                     ktrials=10, test_pct=0.2):
  assert cohort_type in COHORT_TYPE_SET, f'cohort_col must be one of [{COHORT_TYPE_SET}]!'
  assert cohort_type in dashb_df.columns, f'{cohort_type} must exist in dashb_df columns!'
  assert dashb_df['SURG_CASE_KEY'].is_unique, 'Input dashb_df contains duplicated surg_case_key!'
  assert min_cohort_size >= 1, 'min_cohort_size must be a positive int!'
  assert ktrials >= 1, 'ktrials must be a positive int!'

  # Filter data df by thresholding cohort size
  cohort_df = filter_df_by_cohort_count(dashb_df, cohort_type, min_cohort_size)

  # Generate k randomized trials of train-test split, tracking test case surg_case_keys
  kt_test_surg_case_keys = {}
  for kt in range(ktrials):
    # generate each mini set of test case keys independently?
    pass

  return kt_test_surg_case_keys


# Generate k trials of Dataset object for each cohort from k trials of test case keys from modeling_all
def gen_ktrial_cohort2datasets_from_combined_test_case_keys(dashb_df: pd.DataFrame,
                                                            kt_test_case_keys: Dict[int, Iterable],
                                                            cohort_type: str,
                                                            min_cohort_size=50,
                                                            **kwargs):
  assert cohort_type in COHORT_TYPE_SET, f'cohort_col must be one of [{COHORT_TYPE_SET}]!'
  assert cohort_type in dashb_df.columns, f'{cohort_type} must exist in dashb_df columns!'
  assert dashb_df['SURG_CASE_KEY'].is_unique, 'Input dashb_df contains duplicated surg_case_key!'
  assert min_cohort_size >= 1, 'min_cohort_size must be a positive int!'

  # Filter data df by thresholding cohort size
  filtered_df = filter_df_by_cohort_count(dashb_df, cohort_type, min_cohort_size)
  print('Filtered_df case count: ', filtered_df.shape[0])

  # Generate k trials of cohort to dataset mapping
  kt_cohort2datasets = {}
  for kt, test_keys in kt_test_case_keys.items():
    # Ensure all test_keys are in filtered_df, since filtered_df may contain test case keys that are filtered out
    test_keys = np.array(test_keys)[np.in1d(test_keys, filtered_df['SURG_CASE_KEY'])]
    if cohort_type == PRIMARY_PROC_CPTGRP:
      # 1. Create a Dataset object based on each combined test_keys --> get primary_proc_cptgrp column
      # -- keep o2m cases since they might have different hybrid cohort labels!
      dataset = Dataset(filtered_df, test_case_keys=test_keys, remove_o2m=(False, False), **kwargs)
      # 2. Update filtered_df to itself added with primary_proc_cptgrp column
      filtered_df = pd.DataFrame(dataset.cohort_df)

    # Iterate through each cohort in groupby of kt_test_df by cohort_type, construct Dataset from test_case_keys
    cohort2dataset = {}
    effective_cases, cohort_cnt = 0, 0
    for cohort, cohort_df in filtered_df.groupby(cohort_type):
      print(f'Cohort: {cohort}\n\n')
      cohort_cnt += 1
      cohort_test_keys = test_keys[np.in1d(test_keys, cohort_df['SURG_CASE_KEY'])]
      coh_dataset = Dataset(df=cohort_df, test_case_keys=cohort_test_keys, **kwargs)
      # Skip the cohorts that are emptied due to 1. one-to-many cases cleaning 2. nan feature row dropping 3. unseen codes
      if coh_dataset.Xtrain.shape[0] > 0 and coh_dataset.Xtest.shape[0] > 0:
        cohort2dataset[cohort] = coh_dataset
        effective_cases += coh_dataset.Xtrain.shape[0] + coh_dataset.Xtest.shape[0]
    kt_cohort2datasets[kt] = cohort2dataset
    print('=====================================================================================')
    print('Input filtered_df case count: ', filtered_df.shape[0])
    print(f'*Total input cohorts: {cohort_cnt}')
    print(f'*Effective cohort datasets: {len(cohort2dataset)}')
    print(f'*{effective_cases} cases are used out of {dashb_df.shape[0]} available cases')
    print('=====================================================================================')
  return kt_cohort2datasets


# Train a particular type of model on each cohort individually
def train_cohort_clf(md, class_weight, cohort_to_dataset, sda_only=False, surg_only=False, years=None):
  cohort_to_clf = {}
  for cohort, dataset in cohort_to_dataset.items():
    Xtrain, ytrain = dataset.get_Xytrain_by_case_key(dataset.train_case_keys,
                                                     sda_only=sda_only, surg_only=surg_only, years=years)
    clf = get_model_by_cohort(md, class_weight)
    if md == XGBCLF:  # generate a mapping with continuous label class
      xgb_cls = sorted(set(ytrain))
      xgb_cls_to_label = {xgb_cls[i]: i for i in range(len(xgb_cls))}
      ytrain = np.array([xgb_cls_to_label[y] for y in ytrain])

    clf.fit(Xtrain, ytrain)
    cohort_to_clf[cohort] = clf
  return cohort_to_clf


# Train models cohort-wise for k trials
def cohort_modeling_ktrials(decileFtr_config, models: Iterable, kt_cohort2dataset: Dict[int, Iterable],
                            sda_only=False, surg_only=False, years=None):
  # print decile agg funcs
  for k, v in decileFtr_config.items():
    print(k, v)

  models = [LGR, KNN, RMFCLF, XGBCLF] if models is None else models   # GBCLF,
  kt_md2cohort2clf = defaultdict(dict)
  for kt, cohort2dataset in tqdm(kt_cohort2dataset.items()):
    # Fit and eval models
    for md in models:
      print('md=', md)
      cohort2clf = train_cohort_clf(md=md, class_weight=None, cohort_to_dataset=cohort2dataset,
                                    sda_only=sda_only, surg_only=surg_only)
      kt_md2cohort2clf[kt][md] = cohort2clf
  return kt_md2cohort2clf


# Entry point of evaluating k trials of cohort-wise modeling
def cohortwise_modeling_ktrials_eval(models: Iterable,
                                     kt_md2cohort2clf: Dict[int, Dict],
                                     kt_cohort2dataset: Dict[int, Iterable],
                                     sda_only=False, surg_only=False, years=None,
                                     scorers: Iterable[str] = None,
                                     train_perf_df: pd.DataFrame = None, test_perf_df: pd.DataFrame = None):
  for kt, cohort2dataset in tqdm(kt_cohort2dataset.items()):
    for md in models:
      train_perf_df, test_perf_df = eval_cohort_clf(
        cohort2dataset, kt_md2cohort2clf[kt][md], scorers, kt, sda_only=sda_only, surg_only=surg_only, years=years,
        train_perf_df=train_perf_df, test_perf_df=test_perf_df
      )

  return train_perf_df, test_perf_df











# TODO: finish this or discard? Use cases??
def gen_pproc_ktrial_datasets_from_cohortwise_test_case_keys(data_df: pd.DataFrame):

  return


# TODO: use case keys instead of idxs; once done, remove this function!!!
# Generate 'ktrials' of cohort-to-test-indices mapping, save the shuffled dashb_df as well
def gen_ktrial_cohort_test_idxs(dashb_df: pd.DataFrame, cohort_col: str, min_cohort_size=50, ktrials=10, test_pct=0.2):
  assert cohort_col in {SURG_GROUP, PRIMARY_PROC}, f'cohort_col must be one of [{SURG_GROUP, PRIMARY_PROC}]!'
  assert cohort_col in dashb_df.columns, f'{cohort_col} must exist in dashb_df columns!'

  if min_cohort_size is not None:
    assert min_cohort_size >= 0, 'min_cohort_size must be a non-negative int!'
    dashb_df = filter_df_by_cohort_count(dashb_df, cohort_col, min_cohort_size)

  kt_cohort2testIdxs = []  # [ dict{cohort: [idx0, idx1, ...]}, ...]
  kt_pprocDf = []
  for k in range(ktrials):
    # save shuffled df
    pproc_df = dashb_df.sample(frac=1).reset_index(drop=True)
    kt_pprocDf.append(pproc_df)
    # save test indices
    cohort2testIdxs = {}
    for pproc, pp_df in pproc_df.groupby(cohort_col):
      n = pp_df.shape[0]
      cohort2testIdxs[pproc] = np.random.choice(np.arange(n), int(n * test_pct), replace=False)
    kt_cohort2testIdxs.append(cohort2testIdxs)

  return kt_pprocDf, kt_cohort2testIdxs


# TODO: use case keys instead of idxs!!!!!
def gen_pproc_ktrial_datasets(ktrials, decileFtr_config, kt_dfs, kt_cohort2testIdxs, test_pct, min_cohort_size):
  kt_datasets = []
  for k in range(ktrials):
    # Make dataset from existing train-test split cohort idxs
    dataset_k = gen_cohort_to_datasets(
      kt_dfs[k], PRIMARY_PROC, min_cohort_size, kt_cohort2testIdxs[k],
      outcome=NNT, ftr_cols=FEATURE_COLS_NO_WEIGHT_ALLMEDS,
      col2decile_ftrs2aggf=decileFtr_config, onehot_cols=[CCSRS],
      test_pct=test_pct, discretize_cols=['AGE_AT_PROC_YRS'], scaler='robust'
    )
    kt_datasets.append(dataset_k)
  return kt_datasets


# TODO: update this?
def gen_cohort_to_datasets(data_df: pd.DataFrame, cohort_col: str, cohort2test_case_keys: Dict[Hashable, Iterable],
                           min_cohort_size=10, **kwargs) -> Dict[Hashable, Dataset]:
  """
  Generates a mapping from cohort name to a Dataset object, with train & test set splitted.
  Decile-related features are computed only on the cohort cases.

  :param dashb_df: a pd.Dataframe object containing all sources of information
  :param cohort_col: Must be among {SURG_GROUP, PRIMARY_PROC, PRIMARY_PROC_CPTGRP}
  :param min_cohort_size: minimum number of samples per cohort; None if data_df is already filtered
  :param cohort2test_case_keys: a dict mapping cohort name to a numpy array of test case keys to use for that cohort
  :param kwargs: input args for Dataset() instantiation

  :return: A mapping from cohort name to a Dataset object
  """
  # Generate cohort to cohort_data_df mapping
  if min_cohort_size is None:
    cohort_grouped = data_df.groupby(cohort_col)
  else:
    cohort_grouped = filter_df_by_cohort_count(data_df, cohort_col, min_cohort_size).groupby(cohort_col)
  print('[c10 gen_cohort_dataset] Effective sample size: ', cohort_grouped.size().sum())

  # Generate cohort to Dataset() mapping
  cohort_to_dataset = {}
  cases_in_use = 0
  for cohort, cohort_data_df in cohort_grouped:
    print('\n\n***Cohort: ', cohort, '\n\n')
    if cohort2test_case_keys is not None:
      dataset = Dataset(cohort_data_df, test_case_keys=cohort2test_case_keys[cohort], **kwargs)
    else:
      dataset = Dataset(cohort_data_df, **kwargs)  # 'cohort_col' should not exist in onehot_cols in **kwargs

    # Leave out cohorts that are emptied due to 1. one-to-many cases cleaning 2. nan feature row dropping 3. unseen code
    if dataset.Xtrain.shape[0] > 0 and dataset.Xtest.shape[0] > 0:
      cohort_to_dataset[cohort] = dataset
      cases_in_use += dataset.Xtrain.shape[0] + dataset.Xtest.shape[0]

  print('=====================================================================================')
  print(f'*Total input cohorts: {len(cohort_grouped)}')
  print(f'*Actual cohort datasets: {len(cohort_to_dataset)}')
  print(f'*{cases_in_use} cases are used out of {data_df.shape[0]} available cases')
  print('=====================================================================================')
  return cohort_to_dataset




