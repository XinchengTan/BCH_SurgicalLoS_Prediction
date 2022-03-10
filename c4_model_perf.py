import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm import tqdm
from typing import Any, Dict, Iterable

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, make_scorer, roc_auc_score, mean_squared_error
import shap

import utils_plot
from globals import *
from c1_data_preprocessing import Dataset, get_Xys_sda_surg
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
  def classwise_recall(ytrue, ypred, cls):
    # This is not meant to be made a Scorer object in sklearn, but only for evaluation
    if cls not in ytrue:
      return -1
    cls_idxs = np.where(ytrue == cls)[0]
    recall_cls = len((ypred[cls_idxs] != cls)) / len(cls_idxs)
    return recall_cls

  @staticmethod
  def apply_scorers(scorer_names, ytrue, ypred):
    ytrue, ypred = np.array(ytrue), np.array(ypred)
    perf_row_dict = {}
    for scorer_name in scorer_names:
      if scorer_name == SCR_ACC:
        perf_row_dict[scorer_name] = accuracy_score(ytrue, ypred)
      elif scorer_name == SCR_ACC_BAL:
        perf_row_dict[scorer_name] = balanced_accuracy_score(ytrue, ypred)
      elif scorer_name.startswith(SCR_RECALL_PREFIX):
        cls = int(scorer_name.lstrip(SCR_RECALL_PREFIX))
        perf_row_dict[scorer_name] = MyScorer.classwise_recall(ytrue, ypred, cls)
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


def get_default_perf_df(scorers, outcome=NNT):
  columns = ['Trial', 'Xtype', 'Cohort', 'Model', 'Count'] + scorers
  if outcome == NNT:
    return pd.DataFrame(columns=columns + [f'Count_class{c}' for c in NNT_CLASSES])
  else:
    return pd.DataFrame(columns=columns)  # todo: binary class fetch class label & append to 'columns'


def get_class_count(y: Iterable):
  counter = defaultdict(int)
  for cls in NNT_CLASSES:
    counter[f'Count_class{cls}'] = list(y).count(cls)
  return counter


# Display the confusion matrix of one or more models on the dataset where it achieves its median perf across k trials
def show_confmat_of_median_perf_for_mds(perf_df, model_to_confmats, which_md, Xtype, criterion=SCR_ACC):
  which_md = str(which_md)
  if which_md.lower() == 'all':
    for md, kt_to_confmats in model_to_confmats.items():
      show_confmat_of_median_perf_(perf_df, kt_to_confmats, clf2name[md], Xtype, criterion)
  else:
    kt = show_confmat_of_median_perf_(perf_df, model_to_confmats[which_md], which_md, Xtype, criterion)
    if 'Surgeon' in model_to_confmats:
      utils_plot.plot_confusion_matrix(model_to_confmats['Surgeon'][kt], 'Surgeon', Xtype)


# Display the confusion matrix of a particular model on the dataset where it achieves its median perf across k trials
def show_confmat_of_median_perf_(perf_df, kt_to_confmats, md, Xtype, criterion):
  md_name = clf2name[md]
  criterion_sorted = perf_df[perf_df['Model'] == md_name].sort_values(by=criterion).reset_index(drop=True)
  kt = criterion_sorted.iloc[len(kt_to_confmats) // 2]['Trial']
  utils_plot.plot_confusion_matrix(kt_to_confmats[kt], md_name, Xtype)
  return kt


def eval_model_all_ktrials(k_datasets, k_model_dict, eval_by_cohort=SURG_GROUP, eval_sda_only=False, eval_surg_only=False,
                           scorers=None, show_confmat=None, train_perf_df=None, test_perf_df=None):
  model_to_k_confmats_test = defaultdict(dict)  # only for modeling-all, do not support cohort perf yet
  for kt, dataset_k in tqdm(enumerate(k_datasets), total=len(k_datasets)):
    # Model performance
    model_dict = k_model_dict[kt]
    for md, clf in model_dict.items():
      if eval_by_cohort is not None:
        train_perf_df, test_perf_df = eval_model_by_cohort(
          dataset_k, clf, scorers, eval_by_cohort, trial_i=kt, train_perf_df=train_perf_df, test_perf_df=test_perf_df
        )
      else:
        train_perf_df, test_perf_df, confmat_test, surg_confmat_test = eval_model(
          dataset_k, clf, scorers, trial_i=kt, sda_only=eval_sda_only, surg_only=eval_surg_only,
          show_confmat=show_confmat, train_perf_df=train_perf_df, test_perf_df=test_perf_df
        )
        model_to_k_confmats_test[md][kt] = confmat_test
        model_to_k_confmats_test['Surgeon'][kt] = surg_confmat_test

  if show_confmat:
    # show confusion matrix of the median performance
    show_confmat_of_median_perf_for_mds(test_perf_df, model_to_k_confmats_test, show_confmat, Xtype='Test', criterion=SCR_ACC)

  return train_perf_df, test_perf_df


def eval_surgeon_perf(dataset: Dataset, scorers: Iterable, show_confmat):
  train_scores_row_dict, test_scores_row_dict = {}, {}
  if dataset.Xtrain is not None and len(dataset.Xtrain) > 0:
    train_true_preds = dataset.get_surgeon_pred_df_by_case_key(dataset.train_case_keys)
    train_scores_row_dict = MyScorer.apply_scorers(scorers, train_true_preds[dataset.outcome], train_true_preds[SPS_PRED])
  confmat_test = None
  if dataset.Xtest is not None and len(dataset.Xtest) > 0:
    test_true_preds = dataset.get_surgeon_pred_df_by_case_key(dataset.test_case_keys)
    test_scores_row_dict = MyScorer.apply_scorers(scorers, test_true_preds[dataset.outcome], test_true_preds[SPS_PRED])
    if show_confmat:
      confmat_test = confusion_matrix(test_true_preds[dataset.outcome], test_true_preds[SPS_PRED], normalize='true')
  return train_scores_row_dict, test_scores_row_dict, confmat_test


def eval_model_by_cohort(dataset: Dataset, clf, scorers=None, cohort_type=SURG_GROUP, trial_i=None,
                         train_perf_df=None, test_perf_df=None):
  if scorers is None:
    scorers = deepcopy(DEFAULT_SCORERS)

  if train_perf_df is None:
    train_perf_df = get_default_perf_df(scorers)
  if test_perf_df is None:
    test_perf_df = get_default_perf_df(scorers)

  # Get classifier name
  try:
    md_name = clf.model_type
    assert clf.__class__.__name__ == 'SafeOneClassWrapper'
  except AttributeError:
    md_name = clf.__class__.__name__

  if cohort_type not in {SURG_GROUP, PRIMARY_PROC}:  # todo: PRIMARY_PROC_CPTGRP
    raise ValueError('cohort_type must be ont of {SURG_GROUP, PRIMARY_PROC}')

  # TODO: add eval by SDA &/ Surg only
  cohort_to_Xytrain = dataset.get_cohort_to_Xytrains(cohort_type)
  cohort_to_Xytest = dataset.get_cohort_to_Xytests(cohort_type)
  for cohort in cohort_to_Xytrain.keys():
    Xtrain, ytrain = cohort_to_Xytrain[cohort]
    Xtest, ytest = cohort_to_Xytest.get(cohort, (None, None))
    if Xtrain is not None and len(Xtrain) > 0:
      train_pred = clf.predict(Xtrain)
      train_scores = MyScorer.apply_scorers(scorers, ytrain, train_pred)
      train_perf_df = append_perf_row_generic(
        train_perf_df, train_scores, {**get_class_count(ytrain),
                                      **{'Xtype': 'train', 'Cohort': cohort, 'Model': md_name, 'Count': Xtrain.shape[0],
                                         'Trial': trial_i}})
    if Xtest is not None and len(Xtest) > 0:
      test_pred = clf.predict(Xtest)
      test_scores = MyScorer.apply_scorers(scorers, ytest, test_pred)
      test_perf_df = append_perf_row_generic(
        test_perf_df, test_scores, {**get_class_count(ytest),
                                    **{'Xtype': 'test', 'Cohort': cohort, 'Model': md_name, 'Count': Xtest.shape[0],
                                       'Trial': trial_i}})

  train_perf_df.sort_values(by='accuracy', ascending=False, inplace=True)
  test_perf_df.sort_values(by='accuracy', ascending=False, inplace=True)

  return train_perf_df, test_perf_df


# Evaluate model on various groups of cases (e.g. pure SDA cases, cases with surgeon prediction, all cases etc.)
def eval_model(dataset: Dataset, clf, scorers=None, trial_i=None, sda_only=False, surg_only=False, cohort='All',
               show_confmat=False, train_perf_df=None, test_perf_df=None):
  if scorers is None:
    scorers = deepcopy(DEFAULT_SCORERS)
  if train_perf_df is None:
    train_perf_df = get_default_perf_df(scorers)
  if test_perf_df is None:
    test_perf_df = get_default_perf_df(scorers)

  # Get classifier name
  try:
    md_name = clf.model_type
    assert clf.__class__.__name__ == 'SafeOneClassWrapper'
  except AttributeError:
    md_name = clf.__class__.__name__

  # Get train & test Xy
  Xtrain, ytrain, Xtest, ytest = get_Xys_sda_surg(dataset, sda_only, surg_only)

  # Apply trained clf and evaluate
  if Xtrain is not None and len(Xtrain) > 0:
    train_pred = clf.predict(Xtrain)
    train_scores = MyScorer.apply_scorers(scorers, ytrain, train_pred)
    train_perf_df = append_perf_row_generic(
      train_perf_df, train_scores, {**get_class_count(ytrain),
                                    **{'Xtype': 'train', 'Cohort': cohort, 'Model': md_name, 'Trial': trial_i,
                                       'Count': Xtrain.shape[0]}})
  confmat_test = None
  if Xtest is not None and len(Xtest) > 0:
    test_pred = clf.predict(Xtest)
    test_scores = MyScorer.apply_scorers(scorers, ytest, test_pred)
    test_perf_df = append_perf_row_generic(
      test_perf_df, test_scores, {**get_class_count(ytest),
                                  **{'Xtype': 'test', 'Cohort': cohort, 'Model': md_name, 'Trial': trial_i,
                                     'Count': Xtest.shape[0]}})
    if show_confmat:
      confmat_test = confusion_matrix(ytest, test_pred, labels=np.arange(0, MAX_NNT + 2), normalize='true')

  # Surgeon performance
  surg_confmat_test = None
  if surg_only:
    surg_train, surg_test, surg_confmat_test = eval_surgeon_perf(dataset, scorers, show_confmat)
    if len(surg_train) > 0:
      train_perf_df = append_perf_row_generic(
        train_perf_df, surg_train, {**get_class_count(ytrain),
                                    **{'Xtype': 'train', 'Cohort': cohort, 'Model': 'Surgeon', 'Trial': trial_i,
                                       'Count': Xtrain.shape[0]}})
    if len(surg_test) > 0:
      test_perf_df = append_perf_row_generic(
        test_perf_df, surg_test, {**get_class_count(ytest),
                                  **{'Xtype': 'test', 'Cohort': cohort, 'Model': 'Surgeon', 'Trial': trial_i,
                                     'Count': Xtest.shape[0]}})
  return train_perf_df, test_perf_df, confmat_test, surg_confmat_test


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


def summarize_clf_perfs(perf_df: pd.DataFrame, Xtype, sort_by='accuracy'):
  print(f'[{Xtype}] Model performance summary:')
  for col in perf_df.columns:
    if col.startswith('Count'):
      perf_df[col] = pd.to_numeric(perf_df[col])
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


# Format numbers and floats in perf df
def format_perf_df(perf_df: pd.DataFrame):
  formatter = SCR_FORMATTER.copy()
  # format count columns
  for c in NNT_CLASSES:
    formatter[f'Count_class{c}'] = '{:.0f}'.format
    formatter[f'Count_class{c}_mean'] = '{:.0f}'.format
    formatter[f'Count_class{c}_std'] = '{:.0f}'.format
  formatter['Count'] = '{:.0f}'.format
  formatter['Count_mean'] = '{:.0f}'.format
  formatter['Count_std'] = '{:.0f}'.format

  # define actual formatter to be applied
  formatter_ret = deepcopy(formatter)
  for scr in perf_df.columns.to_list():
    if (scr not in {'Model', 'Xtype', 'Cohort', 'Trial'}) and (not scr.startswith('Count')):
      formatter_ret[scr] = formatter[SCR_RMSE] if scr.startswith(SCR_RMSE) else formatter[scr]

  perf_styler = perf_df.style.format(formatter_ret)
  return perf_styler
