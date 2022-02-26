import numpy as np
import pandas as pd
from typing import Dict, Hashable, Any

from globals import *
from c1_data_preprocessing import Dataset
from c4_model_perf import MyScorer, append_perf_row_generic
from c2_models import get_model_by_cohort


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
    print('\n\n***Cohort: ', cohort, '\n\n')
    dataset = Dataset(data_df, **kwargs)  # 'cohort_col' should not exist in onehot_cols in **kwargs
    if dataset.Xtrain.shape[0] > 0 and dataset.Xtest.shape[0] > 0:
      cohort_to_dataset[cohort] = dataset
  return cohort_to_dataset


def train_cohort_clf(md, class_weight, cohort_to_dataset):
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


def eval_cohort_clf(cohort_to_dataset: Dict[str, Dataset], cohort_to_clf: Dict[str, Any], scorers: Dict[str, Any],
                    trial_i=None, train_perf_df=None, test_perf_df=None):
  if scorers is None:
    scorers = MyScorer.get_scorer_dict(DEFAULT_SCORERS)
  if train_perf_df is None:
    train_perf_df = pd.DataFrame(columns=['Trial', 'Xtype', 'Cohort', 'Model'] + list(scorers.keys()))
  if test_perf_df is None:
    test_perf_df = pd.DataFrame(columns=['Trial', 'Xtype', 'Cohort', 'Model'] + list(scorers.keys()))
  for cohort, dataset in cohort_to_dataset.items():
    clf = cohort_to_clf.get(cohort)
    if clf is not None:
      train_pred, test_pred = clf.predict(dataset.Xtrain), clf.predict(dataset.Xtest)
      md_name = clf.__class__.__name__
      if 'XGB' in md_name:
        xgb_cls = sorted(set(dataset.ytrain))
        label_to_xgb_cls = {i: xgb_cls[i] for i in range(len(xgb_cls))}
        train_pred = np.array([label_to_xgb_cls[lb] for lb in train_pred])
        test_pred = np.array([label_to_xgb_cls[lb] for lb in test_pred])
      if not set(train_pred).issubset(set(dataset.ytrain)):
        print(cohort, 'training', set(train_pred), set(dataset.ytrain))
      if not set(test_pred).issubset(set(dataset.ytrain)):
        print('!!!', cohort, 'test', set(test_pred), set(dataset.ytest))
      train_score_dict = MyScorer.apply_scorers(scorers.keys(), dataset.ytrain, train_pred)
      train_perf_df = append_perf_row_generic(
        train_perf_df, train_score_dict, {'Xtype': 'train', 'Cohort': cohort, 'Model': clf.__class__.__name__,
                                          'Count': dataset.Xtrain.shape[0], 'Trial': trial_i})
      test_score_dict = MyScorer.apply_scorers(scorers.keys(), dataset.ytest, test_pred)
      test_perf_df = append_perf_row_generic(
        test_perf_df, test_score_dict, {'Xtype': 'test', 'Cohort': cohort, 'Model': clf.__class__.__name__,
                                        'Count': dataset.Xtest.shape[0], 'Trial': trial_i})
  # Format perf df
  formatter = {scr: SCR_FORMATTER[scr] for scr in scorers.keys()}
  formatter['Count'] = '{:.0f}'.format
  train_styler = train_perf_df.style.format(formatter)
  test_styler = test_perf_df.style.format(formatter)
  return train_perf_df, test_perf_df, train_styler, test_styler


def show_best_clf_per_cohort(perf_df, Xtype):
  best_clf_perf = perf_df.groupby(by=['Cohort', 'Model']).mean().reset_index().groupby('Cohort').max()

  Xsize = best_clf_perf['Count'].sum()
  print(f'Mean {Xtype} size: {Xsize}')
  print(f'Overall mean {Xtype} accuracy: ',
        '{:.2%}'.format(np.dot(best_clf_perf['accuracy'].to_numpy(), best_clf_perf['Count'].to_numpy()) / Xsize))
  formatter = SCR_FORMATTER.copy()
  formatter['Count'] = '{:.0f}'.format
  return best_clf_perf, best_clf_perf.sort_values(by='accuracy', ascending=False).style.format(formatter)
