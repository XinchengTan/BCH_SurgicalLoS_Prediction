import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm import tqdm
from typing import Any, Dict, Iterable

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, make_scorer, roc_auc_score, mean_squared_error
import shap

import utils_plot
from c1_data_preprocessing import Dataset
from globals import *
from utils_eval import *


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
      return -1.0
    cls_idxs = np.where(ytrue == cls)[0]
    recall_cls = sum(np.array(ypred[cls_idxs] == cls)) / len(cls_idxs)
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


# Evaluate models' performance across k trials
def eval_model_all_ktrials(k_datasets, k_model_dict, eval_by_cohort=SURG_GROUP, scorers=None,
                           eval_sda_only=False, eval_surg_only=False, years=None, md_to_show_confmat=None,
                           train_perf_df=None, test_perf_df=None):
  model_to_k_confmats_test = defaultdict(dict)  # only for modeling-all, do not support cohort perf yet
  for kt, dataset_k in tqdm(enumerate(k_datasets), total=len(k_datasets)):
    # Model performance
    model_dict = k_model_dict[kt]
    for md, clf in model_dict.items():
      if eval_by_cohort is not None:
        train_perf_df, test_perf_df = eval_model_by_cohort(
          dataset_k, clf, scorers, eval_by_cohort, trial_i=kt, sda_only=eval_sda_only, surg_only=eval_surg_only,
          years=years, train_perf_df=train_perf_df, test_perf_df=test_perf_df
        )
      else:
        print(md)
        train_perf_df, test_perf_df, confmat_test, surg_confmat_test = eval_model(
          dataset_k, clf, scorers, trial_i=kt, sda_only=eval_sda_only, surg_only=eval_surg_only, years=years,
          show_confmat=md_to_show_confmat, train_perf_df=train_perf_df, test_perf_df=test_perf_df
        )
        model_to_k_confmats_test[md][kt] = confmat_test
        model_to_k_confmats_test[SURGEON][kt] = surg_confmat_test

  print(model_to_k_confmats_test.keys())
  if md_to_show_confmat:
    # show confusion matrix of the median performance
    show_confmat_of_median_perf_for_mds(test_perf_df, model_to_k_confmats_test, md_to_show_confmat, Xtype='Test', criterion=SCR_ACC)

  return train_perf_df, test_perf_df


# Helper function to evaluate surgeon's performance
def eval_surgeon_perf(dataset: Dataset, scorers: Iterable, show_confmat, years=None):
  train_scores_row_dict, test_scores_row_dict = {}, {}
  if dataset.Xtrain is not None and len(dataset.Xtrain) > 0:
    train_true_preds = dataset.get_surgeon_pred_df_by_case_key(dataset.train_case_keys, years=years)
    train_scores_row_dict = MyScorer.apply_scorers(scorers, train_true_preds[dataset.outcome], train_true_preds[SPS_PRED])
  confmat_test = None
  if dataset.Xtest is not None and len(dataset.Xtest) > 0:
    test_true_preds = dataset.get_surgeon_pred_df_by_case_key(dataset.test_case_keys, years=years)
    test_scores_row_dict = MyScorer.apply_scorers(scorers, test_true_preds[dataset.outcome], test_true_preds[SPS_PRED])
    if show_confmat:
      confmat_test = confusion_matrix(test_true_preds[dataset.outcome], test_true_preds[SPS_PRED], normalize='true')
  return train_scores_row_dict, test_scores_row_dict, confmat_test


# Helper function for eval_model_by_cohort()
def eval_model_by_cohort_Xydata(trial_i, dataset: Dataset, clf, cohort_to_XyKeys, Xtype,
                                scorers, perf_df, surg_only, years):
  # Get classifier name
  md_name = get_clf_name(clf)

  # Get year label (None means using all years in dataset)
  year_label = get_year_label(years, dataset)

  # Evaluate on each cohort, include surgeon's performance by request
  for cohort in cohort_to_XyKeys:
    X, y, cohort_case_keys = cohort_to_XyKeys.get(cohort, (np.array([]), np.array([]), np.array([])))
    if len(X) > 0:
      pred = clf.predict(X)
      scores = MyScorer.apply_scorers(scorers, y, pred)
      class_to_counts = get_class_count(y)
      perf_df = append_perf_row_generic(
        perf_df, scores, {**class_to_counts,
                          **{'Xtype': Xtype, 'Cohort': cohort, 'Model': md_name, 'Count': X.shape[0],
                             'Trial': trial_i, 'Year': year_label}
                          })
      if surg_only:
        true_surg_preds = dataset.get_surgeon_pred_df_by_case_key(cohort_case_keys, years=years)
        scores_surg = MyScorer.apply_scorers(scorers, true_surg_preds[dataset.outcome], true_surg_preds[SPS_PRED])
        perf_df = append_perf_row_generic(
          perf_df, scores_surg, {**class_to_counts,
                                 **{'Xtype': Xtype, 'Cohort': cohort, 'Model': SURGEON, 'Count': X.shape[0],
                                    'Trial': trial_i, 'Year': year_label}
                                 })
  return perf_df


# Evaluates the model performance on each cohort
def eval_model_by_cohort(dataset: Dataset, clf, scorers=None, cohort_type=SURG_GROUP, trial_i=None,
                         sda_only=False, surg_only=False, years=None,
                         train_perf_df=None, test_perf_df=None):
  assert cohort_type in COHORT_TYPE_SET, f'cohort_type must be ont of {COHORT_TYPE_SET}'

  if scorers is None:
    scorers = deepcopy(DEFAULT_SCORERS)
  if train_perf_df is None:
    train_perf_df = get_default_perf_df(scorers)
  if test_perf_df is None:
    test_perf_df = get_default_perf_df(scorers)

  # Evaluate on Training set for each cohort
  cohort_to_XyKeys_train = dataset.get_cohort_to_Xytrains(cohort_type, sda_only=sda_only, surg_only=surg_only, years=years)
  train_perf_df = eval_model_by_cohort_Xydata(trial_i, dataset, clf, cohort_to_XyKeys_train, 'train', scorers=scorers,
                                              perf_df=train_perf_df, surg_only=surg_only, years=years)

  # Evaluate on Test set for each cohort
  cohort_to_XyKeys_test = dataset.get_cohort_to_Xytests(cohort_type, sda_only=sda_only, surg_only=surg_only, years=years)
  test_perf_df = eval_model_by_cohort_Xydata(trial_i, dataset, clf, cohort_to_XyKeys_test, 'test', scorers=scorers,
                                             perf_df=test_perf_df, surg_only=surg_only, years=years)

  train_perf_df.sort_values(by=['Count', 'Cohort', 'Model'], ascending=False, inplace=True)  # 'accuracy'
  test_perf_df.sort_values(by=['Count', 'Cohort', 'Model'], ascending=False, inplace=True)  # 'accuracy'
  return train_perf_df, test_perf_df


# Evaluate model on various groups of cases (e.g. pure SDA cases, cases with surgeon prediction, all cases etc.)
def eval_model(dataset: Dataset, clf, scorers=None, trial_i=None, sda_only=False, surg_only=False, years=None,
               cohort='All', show_confmat=False, train_perf_df=None, test_perf_df=None):
  if scorers is None:
    scorers = deepcopy(DEFAULT_SCORERS)
  if train_perf_df is None:
    train_perf_df = get_default_perf_df(scorers)
  if test_perf_df is None:
    test_perf_df = get_default_perf_df(scorers)

  # Get classifier name
  md_name = get_clf_name(clf)

  # Get year label (None means using all years in dataset)
  year_label = get_year_label(years, dataset)

  # Get train & test X, y under sda, surg, years filters
  Xtrain, ytrain = dataset.get_Xytrain_by_case_key(dataset.train_case_keys,
                                                   sda_only=sda_only, surg_only=surg_only, years=years)
  Xtest, ytest = dataset.get_Xytest_by_case_key(dataset.test_case_keys,
                                                sda_only=sda_only, surg_only=surg_only, years=years)

  # Apply trained clf and evaluate
  if Xtrain is not None and len(Xtrain) > 0:
    train_pred = clf.predict(Xtrain)
    train_scores = MyScorer.apply_scorers(scorers, ytrain, train_pred)
    train_perf_df = append_perf_row_generic(
      train_perf_df, train_scores, {**get_class_count(ytrain),
                                    **{'Xtype': 'train', 'Cohort': cohort, 'Model': md_name, 'Trial': trial_i,
                                       'Count': Xtrain.shape[0], 'Year': year_label}})
  confmat_test = None
  if Xtest is not None and len(Xtest) > 0:
    test_pred = clf.predict(Xtest)
    test_scores = MyScorer.apply_scorers(scorers, ytest, test_pred)
    test_perf_df = append_perf_row_generic(
      test_perf_df, test_scores, {**get_class_count(ytest),
                                  **{'Xtype': 'test', 'Cohort': cohort, 'Model': md_name, 'Trial': trial_i,
                                     'Count': Xtest.shape[0], 'Year': year_label}})
    if show_confmat:
      confmat_test = confusion_matrix(ytest, test_pred, labels=np.arange(0, MAX_NNT + 2), normalize='true')

  # Surgeon performance
  surg_confmat_test = None
  if surg_only:
    surg_train, surg_test, surg_confmat_test = eval_surgeon_perf(dataset, scorers, show_confmat, years=years)
    if len(surg_train) > 0:
      train_perf_df = append_perf_row_generic(
        train_perf_df, surg_train, {**get_class_count(ytrain),
                                    **{'Xtype': 'train', 'Cohort': cohort, 'Model': SURGEON, 'Trial': trial_i,
                                       'Count': Xtrain.shape[0], 'Year': year_label}})
    if len(surg_test) > 0:
      test_perf_df = append_perf_row_generic(
        test_perf_df, surg_test, {**get_class_count(ytest),
                                  **{'Xtype': 'test', 'Cohort': cohort, 'Model': SURGEON, 'Trial': trial_i,
                                     'Count': Xtest.shape[0], 'Year': year_label}})
  return train_perf_df, test_perf_df, confmat_test, surg_confmat_test


# Summarize the classifiers' performance for each year individually
def summarize_clf_perfs(perf_df: pd.DataFrame, Xtype, sort_by=['accuracy_mean']):
  print(f'[{Xtype}] Model performance summary:')
  perf_df = to_numeric_count_cols(perf_df)

  # Group by md, aggregate across trial
  clf_perfs = pd.merge(perf_df.groupby(by=['Model', 'Cohort', 'Year']).mean().reset_index(),
                       perf_df.groupby(by=['Model', 'Cohort', 'Year']).std().reset_index(),
                       on=['Model', 'Cohort', 'Year'],
                       how='left',
                       suffixes=('_mean', '_std')) \
    .dropna(axis=1) \
    .sort_values(by=sort_by, ascending=False) \
    .reset_index(drop=True)
  clf_perfs_styler = format_perf_df(clf_perfs)
  return clf_perfs, clf_perfs_styler
