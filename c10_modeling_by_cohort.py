import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, Hashable, List, Iterable

from globals import *
from c1_data_preprocessing import Dataset
from c4_model_perf import MyScorer, append_perf_row_generic, format_perf_df
from c2_models import get_model_by_cohort, SafeOneClassWrapper
from c9_cohort_modeling_perf import eval_cohort_clf


# helper function to filter dashb_df by cohort count
def filter_df_by_cohort_count(dashb_df: pd.DataFrame,  cohort_col: str, min_cohort_size: int):
  cohort_groupby = dashb_df.groupby(cohort_col)
  if min_cohort_size > 1:
    filtered_cohort = cohort_groupby.filter(lambda x: len(x) >= min_cohort_size)
    return filtered_cohort
  return dashb_df


# TODO: use case keys instead of idxs!!!!!
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


def gen_pproc_ktrial_datasets_from_test_case_keys(data_df: pd.DataFrame, kt_test_case_keys: List[Iterable]):
  kt_datasets = []
  for kt, test_case_keys in enumerate(kt_test_case_keys):
    pass
  return


# TODO: use case keys instead of idxs!!!!!
def gen_pproc_ktrial_datasets(ktrials, decileFtr_config, kt_dfs, kt_cohort2testIdxs, test_pct, min_cohort_size):
  kt_datasets = []
  for k in range(ktrials):
    # Make dataset from existing train-test split cohort idxs
    dataset_k = gen_cohort_datasets(
      kt_dfs[k], PRIMARY_PROC, min_cohort_size, kt_cohort2testIdxs[k],
      outcome=NNT, ftr_cols=FEATURE_COLS_NO_WEIGHT_ALLMEDS,
      col2decile_ftrs2aggf=decileFtr_config, onehot_cols=[CCSRS],
      test_pct=test_pct, discretize_cols=['AGE_AT_PROC_YRS'], scaler='robust'
    )
    kt_datasets.append(dataset_k)
  return kt_datasets


# TODO: use case keys instead of idxs!!!!!
def gen_cohort_datasets(dashb_df: pd.DataFrame, cohort_col: str, min_cohort_size=10, cohort2testIdxs=None,
                        **kwargs) -> Dict[Hashable, Dataset]:
  """
  Generates a mapping from cohort name to a Dataset object, with train & test set splitted.
  Decile-related features are computed only on the cohort cases.

  :param dashb_df: a pd.Dataframe object containing all sources of information
  :param cohort_col: Either SURG_GROUP or PRIMARY_PROC
  :param min_cohort_size: minimum number of samples per cohort
  :param cohort2testIdxs: a dict mapping cohort name to a numpy array of test indices to use for that cohort
  :param kwargs: input args for Dataset() instantiation

  :return: A mapping from cohort name to a Dataset object
  """
  assert cohort_col in {SURG_GROUP, PRIMARY_PROC}, f'cohort_col must be one of [{SURG_GROUP, PRIMARY_PROC}]!'
  assert cohort_col in dashb_df.columns, f'{cohort_col} must exist in dashb_df columns!'

  # Generate cohort to cohort_data_df mapping
  cohort_groupby = filter_df_by_cohort_count(dashb_df, cohort_col, min_cohort_size).groupby(cohort_col)
  print('[c10 gen_cohort_dataset] Effective sample size: ', cohort_groupby.size().sum())

  # Generate cohort to Dataset() mapping
  cohort_to_dataset = {}
  cases_in_use = 0
  for cohort, cohort_data_df in cohort_groupby:
    print('\n\n***Cohort: ', cohort, '\n\n')
    if cohort2testIdxs is not None:
      dataset = Dataset(cohort_data_df, test_idxs=cohort2testIdxs[cohort], **kwargs)
    else:
      dataset = Dataset(cohort_data_df, **kwargs)  # 'cohort_col' should not exist in onehot_cols in **kwargs

    # Leave out cohorts that are emptied due to 1. one-to-many cases cleaning 2. nan feature row dropping 3. unseen code
    if dataset.Xtrain.shape[0] > 0 and dataset.Xtest.shape[0] > 0:
      cohort_to_dataset[cohort] = dataset
      cases_in_use += dataset.Xtrain.shape[0] + dataset.Xtest.shape[0]

  print('=====================================================================================')
  print(f'*Total input cohorts: {len(cohort_groupby)}')
  print(f'*Actual cohort datasets: {len(cohort_to_dataset)}')
  print(f'*{cases_in_use} cases are used out of {dashb_df.shape[0]} available cases')
  print('=====================================================================================')
  return cohort_to_dataset


def train_cohort_clf(md, class_weight, cohort_to_dataset, sda_only=False, surg_only=False, years=None):
  # TODO: add training only on SDA cases
  cohort_to_clf = {}
  for cohort, dataset in cohort_to_dataset.items():
    clf = get_model_by_cohort(md, class_weight)
    if md == XGBCLF:  # generate a mapping with continuous label class
      xgb_cls = sorted(set(dataset.ytrain))
      xgb_cls_to_label = {xgb_cls[i]: i for i in range(len(xgb_cls))}
      ytrain = np.array([xgb_cls_to_label[y] for y in dataset.ytrain])
    else:
      ytrain = np.array(dataset.ytrain)
    clf.fit(dataset.Xtrain, ytrain)
    cohort_to_clf[cohort] = clf
  return cohort_to_clf


def cohort_modeling_ktrials(decileFtr_config, models, k_datasets, train_perf_df=None, test_perf_df=None):
  # print decile agg funcs
  for k, v in decileFtr_config.items():
    print(k, v)

  models = [LGR, KNN, RMFCLF, XGBCLF] if models is None else models  # GBCLF,
  for k in tqdm(range(len(k_datasets))):
    # Fit and eval models
    for md in models:
      print('md=', md)
      cohort_to_clf_pproc = train_cohort_clf(md=md, class_weight=None, cohort_to_dataset=k_datasets[k])
      train_perf_df, test_perf_df = eval_cohort_clf(
        k_datasets[k], cohort_to_clf_pproc, None, k, train_perf_df, test_perf_df
      )

  return train_perf_df, test_perf_df


