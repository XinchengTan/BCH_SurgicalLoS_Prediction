import numpy as np
import pandas as pd
from typing import Dict, Hashable, Any

from globals import *
from c1_data_preprocessing import Dataset
from c4_model_perf import MyScorer, append_perf_row_generic
from c2_models import get_model


def gen_cohort_datasets(dashb_df: pd.DataFrame, cohort_col: str, min_cohort_size=10, **kwargs) -> Dict[Hashable, Dataset]:
  """
  Generates a mapping from cohort name to a Dataset object, with train & test set splitted.
  Decile-related features are computed only on the cohort cases.

  :param dashb_df: a pd.Dataframe object containing all sources of information
  :param cohort_col: Either SURG_GROUP or PRIMARY_PROC
  :param min_cohort_size: minimum number of samples per cohort
  :param kwargs: input args for Dataset() instantiation

  :return: A mapping from cohort name to a Dataset object
  """
  assert cohort_col in {SURG_GROUP, PRIMARY_PROC}, f'cohort_col must be one of [{SURG_GROUP, PRIMARY_PROC}]!'
  assert cohort_col in dashb_df.columns, f'{cohort_col} must exist in dashb_df columns!'
  assert min_cohort_size >= 0, 'min_cohort_size must be a non-negative int!'

  # Generate cohort to data_df mapping
  cohort_groupby = dashb_df.groupby(cohort_col)
  if min_cohort_size > 1:
    cohort_groupby = cohort_groupby.filter(lambda x: len(x) >= min_cohort_size).groupby(cohort_col)
  print('[c11 gen_cohort_dataset] Effective sample size: ', cohort_groupby.size().sum())

  # Generate cohort to Dataset() mapping
  cohort_to_dataset = {}
  for cohort, data_df in cohort_groupby:
    cohort_to_dataset[cohort] = Dataset(data_df, **kwargs)  # 'cohort_col' should not exist in onehot_cols in **kwargs

  return cohort_to_dataset


def train_cohort_models(md, class_weight, cohort_to_dataset):
  cohort_to_clf = {}
  for cohort, dataset in cohort_to_dataset:
    clf = get_model(md, class_weight)
    clf.fit(dataset.Xtrain, dataset.ytrain)
    cohort_to_clf[cohort] = clf
  return cohort_to_clf


def eval_cohort_models(cohort_to_dataset: Dict[str, Dataset], cohort_to_clf: Dict[str, Any], scorers: Dict[str, Any]):
  train_perf_df = pd.DataFrame(columns=['Xtype', 'Cohort', 'Model'] + list(scorers.keys()))
  test_perf_df = train_perf_df.copy()
  for cohort, dataset in cohort_to_dataset.items():
    clf = cohort_to_clf.get(cohort)
    if clf is not None:
      train_pred, test_pred = clf.predict(dataset.Xtrain), clf.predict(dataset.Xtest)
      train_score_dict = MyScorer.apply_scorers(scorers.keys(), dataset.ytrain, train_pred)
      train_perf_df = append_perf_row_generic(train_perf_df, train_score_dict, {'Xtype': 'train', 'Cohort': cohort,
                                                                                'Model': clf.__class__.__name__})
      test_score_dict = MyScorer.apply_scorers(scorers.keys(), dataset.ytest, test_pred)
      test_perf_df = append_perf_row_generic(test_perf_df, test_score_dict, {'Xtype': 'test', 'Cohort': cohort,
                                                                             'Model': clf.__class__.__name__})
  return train_perf_df, test_perf_df

