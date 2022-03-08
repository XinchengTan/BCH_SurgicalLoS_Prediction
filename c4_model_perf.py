import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Dict, Any
from tqdm import tqdm

from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer, roc_auc_score, mean_squared_error
import shap

from globals import *
from c1_data_preprocessing import Dataset
from c2_models import SafeOneClassWrapper


class MyScorer:

  def __init__(self):
    return

  @staticmethod
  def get_scorer_dict(scorer_names):
    scr_dict = {}
    for scorer in scorer_names:
      if scorer == SCR_ACC:
        scr_dict[scorer] = 'accuracy'
      elif scorer == SCR_ACC_BAL:
        scr_dict[scorer] = 'balanced_accuracy'
      elif scorer == SCR_AUC:
        scr_dict[scorer] = 'roc_auc'
      elif scorer == SCR_RMSE:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_rmse, greater_is_better=False)
      elif scorer == SCR_ACC_ERR1:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_1nnt_tol, greater_is_better=True)
      elif scorer == SCR_ACC_ERR2:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_2nnt_tol, greater_is_better=True)
      elif scorer == SCR_OVERPRED:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_overpred_pct, greater_is_better=False)
      elif scorer == SCR_UNDERPRED:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_underpred_pct, greater_is_better=False)
      else:
        raise Warning(f"Scorer {scorer} is not supported yet!")
    return scr_dict

  @staticmethod
  def scorer_rmse(ytrue, ypred):
    mse = mean_squared_error(ytrue, ypred)
    return np.sqrt(mse)

  @staticmethod
  def scorer_1nnt_tol(ytrue, ypred):
    # accuracy within +-1 nnt error tolerance
    acc_1nnt_tol = len(np.where(np.abs(ytrue - ypred) <= 1)[0]) / len(ytrue)
    return acc_1nnt_tol

  @staticmethod
  def scorer_2nnt_tol(ytrue, ypred):
    # accuracy within +-1 nnt error tolerance
    acc_1nnt_tol = len(np.where(np.abs(ytrue - ypred) <= 2)[0]) / len(ytrue)
    return acc_1nnt_tol

  @staticmethod
  def scorer_overpred_pct(ytrue, ypred):
    overpred_pct = len(np.where((ypred - ytrue) > 2)[0]) / len(ytrue)
    return overpred_pct

  @staticmethod
  def scorer_underpred_pct(ytrue, ypred):
    underpred_pct = len(np.where((ytrue - ypred) > 2)[0]) / len(ytrue)
    return underpred_pct

  @staticmethod
  def apply_scorers(scorer_names, ytrue, ypred):
    perf_row_dict = {}
    for scorer_name in scorer_names:
      if scorer_name == SCR_ACC:
        perf_row_dict[scorer_name] = accuracy_score(ytrue, ypred)
      elif scorer_name == SCR_ACC_BAL:
        perf_row_dict[scorer_name] = balanced_accuracy_score(ytrue, ypred)
      elif scorer_name == SCR_RMSE:
        perf_row_dict[scorer_name] = MyScorer.scorer_rmse(ytrue, ypred)
      elif scorer_name == SCR_ACC_ERR1:
        perf_row_dict[scorer_name] = MyScorer.scorer_1nnt_tol(ytrue, ypred)
      elif scorer_name == SCR_ACC_ERR2:
        perf_row_dict[scorer_name] = MyScorer.scorer_2nnt_tol(ytrue, ypred)
      elif scorer_name == SCR_OVERPRED:
        perf_row_dict[scorer_name] = MyScorer.scorer_overpred_pct(ytrue, ypred)
      elif scorer_name == SCR_UNDERPRED:
        perf_row_dict[scorer_name] = MyScorer.scorer_underpred_pct(ytrue, ypred)
      elif scorer_name == SCR_AUC:
        perf_row_dict[scorer_name] = None  # TODO: find a better solution to handle this
        # perf_row_dict[scorer_name] = roc_auc_score(ytrue, ypred)  # ypred is actually yscore: clf.predict_proba(X)[:, 1]
      else:
        raise NotImplementedError('%s not implemented' % scorer_name)
    return perf_row_dict


def eval_surgeon_perf(dataset: Dataset, scorers):
  train_scores_row_dict, test_scores_row_dict = {}, {}
  if dataset.Xtrain:
    train_surg_pred = dataset.get_surgeon_pred_df_by_case_key(dataset.train_case_keys)[SPS_PRED]
    train_scores_row_dict = MyScorer.apply_scorers(scorers, dataset.ytrain, train_surg_pred)
  if dataset.Xtest:
    test_surg_pred = dataset.get_surgeon_pred_df_by_case_key(dataset.test_case_keys)[SPS_PRED]
    test_scores_row_dict = MyScorer.apply_scorers(scorers, dataset.ytest, test_surg_pred)
  return train_scores_row_dict, test_scores_row_dict


def eval_model_all_ktrials(k_datasets, k_model_dict, eval_by_cohort=SURG_GROUP, eval_sda_only=False,
                           train_perf_df=None, test_perf_df=None):
  for kt, dataset_k in tqdm(enumerate(k_datasets)):
    model_dict = k_model_dict[kt]
    for md, clf in model_dict.items():
      if eval_sda_only:
        train_perf_df, test_perf_df = eval_model_sda_only(
          dataset_k, clf, trial_i=kt, train_perf_df_sda=train_perf_df, test_perf_df_sda=test_perf_df
        )
      else:
        train_perf_df, test_perf_df = eval_model(
          dataset_k, clf, by_cohort=eval_by_cohort, trial_i=kt, train_perf_df=train_perf_df, test_perf_df=test_perf_df
        )

  return train_perf_df, test_perf_df


def summarize_clf_perfs(perf_df: pd.DataFrame, Xtype, sort_by='accuracy'):
  print(f'[{Xtype}] Model performance summary:')
  # Group by md, aggregate across trial
  clf_perfs = pd.merge(perf_df.groupby(by=['Model', 'Cohort']).mean().reset_index(),
                       perf_df.groupby(by=['Model', 'Cohort']).std().reset_index(),
                       on=['Model', 'Cohort'],
                       how='left',
                       suffixes=('_mean', '_std')) \
    .dropna(axis=1) \
    .sort_values(by=sort_by+'_mean', ascending=False)
  clf_perfs_styler = format_perf_df(clf_perfs)
  return clf_perfs, clf_perfs_styler


def eval_model(dataset: Dataset, clf, scorers=None, by_cohort=None, trial_i=None,
               train_perf_df=None, test_perf_df=None):
  if scorers is None:
    scorers = MyScorer.get_scorer_dict(DEFAULT_SCORERS)
  if train_perf_df is None:
    train_perf_df = pd.DataFrame(columns=['Trial', 'Xtype', 'Cohort', 'Model'] + list(scorers.keys()))
  if test_perf_df is None:
    test_perf_df = pd.DataFrame(columns=['Trial', 'Xtype', 'Cohort', 'Model'] + list(scorers.keys()))
  md_name = clf.__class__.__name__ if not isinstance(clf, SafeOneClassWrapper) else clf.model_type

  if by_cohort == SURG_GROUP or by_cohort == PRIMARY_PROC:
    cohort_to_Xytrain = dataset.get_cohort_to_Xytrains(by_cohort)
    cohort_to_Xytest = dataset.get_cohort_to_Xytests(by_cohort)
    for cohort in cohort_to_Xytrain.keys():
      Xtrain, ytrain = cohort_to_Xytrain[cohort]
      Xtest, ytest = cohort_to_Xytest.get(cohort, (None, None))
      train_pred = clf.predict(Xtrain)
      train_scores = MyScorer.apply_scorers(scorers.keys(), ytrain, train_pred)
      train_perf_df = append_perf_row_generic(
        train_perf_df, train_scores, {'Xtype': 'train', 'Cohort': cohort, 'Model': md_name,
                                      'Count': Xtrain.shape[0], 'Trial': trial_i})
      if Xtest is not None:
        test_pred = clf.predict(Xtest)
        test_scores = MyScorer.apply_scorers(scorers.keys(), ytest, test_pred)
        test_perf_df = append_perf_row_generic(
          test_perf_df, test_scores, {'Xtype': 'test', 'Cohort': cohort, 'Model': md_name,
                                      'Count': Xtest.shape[0], 'Trial': trial_i})
  else:
    train_pred, test_pred = clf.predict(dataset.Xtrain), clf.predict(dataset.Xtest)
    train_scores = MyScorer.apply_scorers(scorers.keys(), dataset.ytrain, train_pred)
    test_scores = MyScorer.apply_scorers(scorers.keys(), dataset.ytest, test_pred)
    train_perf_df = append_perf_row_generic(
      train_perf_df, train_scores, {'Xtype': 'train', 'Cohort': 'All', 'Model': md_name, 'Trial': trial_i})
    test_perf_df = append_perf_row_generic(
      test_perf_df, test_scores, {'Xtype': 'test', 'Cohort': 'All', 'Model': md_name, 'Trial': trial_i})

  train_perf_df.sort_values(by='accuracy', ascending=False, inplace=True)
  test_perf_df.sort_values(by='accuracy', ascending=False, inplace=True)

  return train_perf_df, test_perf_df


# Evaluate model only on SDA cases
def eval_model_sda_only(dataset: Dataset, clf, scorers=None, trial_i=None,
                        train_perf_df_sda=None, test_perf_df_sda=None):
  if scorers is None:
    scorers = MyScorer.get_scorer_dict(DEFAULT_SCORERS)
  if train_perf_df_sda is None:
    train_perf_df_sda = pd.DataFrame(columns=['Trial', 'Xtype', 'Cohort', 'Model'] + list(scorers.keys()))
  if test_perf_df_sda is None:
    test_perf_df_sda = pd.DataFrame(columns=['Trial', 'Xtype', 'Cohort', 'Model'] + list(scorers.keys()))
  md_name = clf.__class__.__name__ if not isinstance(clf, SafeOneClassWrapper) else clf.model_type

  # Get Xytrain_sda, Xytest_sda
  Xtrain_sda, ytrain_sda = dataset.get_sda_Xytrain()
  Xtest_sda, ytest_sda = dataset.get_sda_Xytest()

  # Apply trained clf and evaluate
  train_pred, test_pred = clf.predict(Xtrain_sda), clf.predict(Xtest_sda)
  train_scores = MyScorer.apply_scorers(scorers.keys(), ytrain_sda, train_pred)
  test_scores = MyScorer.apply_scorers(scorers.keys(), ytest_sda, test_pred)
  train_perf_df_sda = append_perf_row_generic(
    train_perf_df_sda, train_scores, {'Xtype': 'train', 'Cohort': 'All', 'Model': md_name, 'Trial': trial_i})
  test_perf_df_sda = append_perf_row_generic(
    test_perf_df_sda, test_scores, {'Xtype': 'test', 'Cohort': 'All', 'Model': md_name, 'Trial': trial_i})
  # TODO: add eval by cohort
  return train_perf_df_sda, test_perf_df_sda


def append_perf_row_generic(perf_df, score_dict: Dict, info_col_dict: Dict[str, Any]):
  score_dict.update(info_col_dict)
  perf_df = perf_df.append(score_dict, ignore_index=True)
  return perf_df


def append_perf_row(perf_df, trial, md, scores_row_dict: Dict):
  scores_row_dict['Model'] = md
  scores_row_dict['Trial'] = trial
  perf_df = perf_df.append(scores_row_dict, ignore_index=True)
  return perf_df


def append_perf_row_surg(surg_perf_df: pd.DataFrame, trial, scores_row_dict):
  if scores_row_dict is None or len(scores_row_dict) == 0:
    return surg_perf_df
  scores_row_dict['Trial'] = trial
  scores_row_dict['Model'] = 'Surgeon-train'
  surg_perf_df = surg_perf_df.append(scores_row_dict, ignore_index=True)
  return surg_perf_df


# Format numbers and floats in perf df
def format_perf_df(perf_df: pd.DataFrame):
  formatter = SCR_FORMATTER.copy()
  formatter['Count'] = '{:.0f}'.format
  formatter_ret = deepcopy(formatter)
  for k, v in formatter.items():
    formatter_ret[k+'_mean'] = v
    formatter_ret[k+'_std'] = v
  perf_styler = perf_df.style.format(formatter_ret)
  return perf_styler


