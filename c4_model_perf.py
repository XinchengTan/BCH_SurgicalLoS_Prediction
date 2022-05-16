import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm import tqdm
from typing import Any, Dict, Iterable

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, make_scorer, \
  mean_absolute_error, mean_squared_error, recall_score, precision_score, f1_score
from sklearn.metrics import auc, roc_auc_score, plot_roc_curve, precision_recall_curve, plot_precision_recall_curve
import shap

import utils_plot
from c1_data_preprocessing import Dataset
from globals import *
from globals_fs import *
from utils_eval import *


class MyScorer:

  def __init__(self):
    return

  @staticmethod
  def get_scorer_dict(scorer_names, binary_cls=True):
    scr_dict = {}
    for scorer in scorer_names:
      if scorer == SCR_ACC:
        scr_dict[scorer] = 'accuracy'
      elif scorer == SCR_ACC_BAL:
        scr_dict[scorer] = 'balanced_accuracy'
      elif scorer == SCR_AUROC:
        scr_dict[scorer] = 'roc_auc' if binary_cls else 'roc_auc_ovr'
      elif scorer == SCR_RECALL_BINCLF:
        scr_dict[scorer] = 'recall'
      elif scorer == SCR_PREC_BINCLF:
        scr_dict[scorer] = 'precision'
      elif scorer == SCR_F1_BINCLF:
        scr_dict[scorer] = 'f1'
      elif scorer == SCR_RECALL_BINCLF_NEG:
        scr_dict[scorer] = make_scorer(recall_score, pos_label=0)
      elif scorer == SCR_PREC_BINCLF_NEG:
        scr_dict[scorer] = make_scorer(precision_score, pos_label=0)
      elif scorer == SCR_F1_BINCLF_NEG:
        scr_dict[scorer] = make_scorer(f1_score, pos_label=0)
      elif scorer == SCR_MAE:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_mae, greater_is_better=False)
      elif scorer == SCR_RMSE:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_rmse, greater_is_better=False)
      elif scorer == SCR_ACC_ERR1:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_1nnt_tol, greater_is_better=True)
      elif scorer == SCR_ACC_ERR2:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_2nnt_tol, greater_is_better=True)
      elif scorer == SCR_OVERPRED2:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_overpred_pct, greater_is_better=False)
      elif scorer == SCR_UNDERPRED2:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_underpred_pct, greater_is_better=False)
      elif scorer == SCR_OVERPRED0:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_overpred_pct0, greater_is_better=False)
      elif scorer == SCR_UNDERPRED0:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_underpred_pct0, greater_is_better=False)
      else:
        raise Warning(f"Scorer {scorer} is not supported yet!")
    return scr_dict

  @staticmethod
  def scorer_mae(ytrue, ypred):
    mae = mean_absolute_error(ytrue, ypred)
    return mae

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
  def scorer_overpred_pct0(ytrue, ypred):
    overpred_pct = len(np.where((ypred - ytrue) > 0)[0]) / len(ytrue)
    return overpred_pct

  @staticmethod
  def scorer_underpred_pct0(ytrue, ypred):
    underpred_pct = len(np.where((ytrue - ypred) > 0)[0]) / len(ytrue)
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
  def apply_scorers(scorer_names, ytrue, ypred, enable_warning=True):
    ytrue, ypred = np.array(ytrue), np.array(ypred)
    perf_row_dict = {}
    for scorer_name in scorer_names:
      if scorer_name == SCR_ACC:
        perf_row_dict[scorer_name] = accuracy_score(ytrue, ypred)
      elif scorer_name == SCR_ACC_BAL:
        perf_row_dict[scorer_name] = balanced_accuracy_score(ytrue, ypred)
      elif scorer_name.startswith(SCR_RECALL_PREFIX):
        cls = int(scorer_name.lstrip(SCR_RECALL_PREFIX))
        if COMBINE_01:
          cls -= 1  # map to class label in modeling
        perf_row_dict[scorer_name] = MyScorer.classwise_recall(ytrue, ypred, cls)
      elif scorer_name == SCR_MAE:
        perf_row_dict[scorer_name] = MyScorer.scorer_mae(ytrue, ypred)
      elif scorer_name == SCR_RMSE:
        perf_row_dict[scorer_name] = MyScorer.scorer_rmse(ytrue, ypred)
      elif scorer_name == SCR_ACC_ERR1:
        perf_row_dict[scorer_name] = MyScorer.scorer_1nnt_tol(ytrue, ypred)
      elif scorer_name == SCR_ACC_ERR2:
        perf_row_dict[scorer_name] = MyScorer.scorer_2nnt_tol(ytrue, ypred)
      elif scorer_name == SCR_OVERPRED2:
        perf_row_dict[scorer_name] = MyScorer.scorer_overpred_pct(ytrue, ypred)
      elif scorer_name == SCR_UNDERPRED2:
        perf_row_dict[scorer_name] = MyScorer.scorer_underpred_pct(ytrue, ypred)
      elif scorer_name == SCR_OVERPRED0:
        perf_row_dict[scorer_name] = MyScorer.scorer_overpred_pct0(ytrue, ypred)
      elif scorer_name == SCR_UNDERPRED0:
        perf_row_dict[scorer_name] = MyScorer.scorer_underpred_pct0(ytrue, ypred)
      elif scorer_name == SCR_RECALL_BINCLF:
        perf_row_dict[scorer_name] = recall_score(ytrue, ypred)
      elif scorer_name == SCR_PREC_BINCLF:
        perf_row_dict[scorer_name] = precision_score(ytrue, ypred)
      elif scorer_name == SCR_F1_BINCLF:
        perf_row_dict[scorer_name] = f1_score(ytrue, ypred)
      elif scorer_name == SCR_RECALL_BINCLF_NEG:
        perf_row_dict[scorer_name] = recall_score(ytrue, ypred, pos_label=0)
      elif scorer_name == SCR_PREC_BINCLF_NEG:
        perf_row_dict[scorer_name] = precision_score(ytrue, ypred, pos_label=0)
      elif scorer_name == SCR_F1_BINCLF_NEG:
        perf_row_dict[scorer_name] = f1_score(ytrue, ypred, pos_label=0)
      elif scorer_name == SCR_AUPRC:
        perf_row_dict[scorer_name] = -1
      elif scorer_name == SCR_AUROC_NEG:
        perf_row_dict[scorer_name] = -1
      elif scorer_name == SCR_AUROC:
        perf_row_dict[scorer_name] = -1
        if enable_warning:
          warnings.warn('Default ROC AUC to -1. To calculate actual AUC, please use MyScorer.calc_auc_roc()')
      else:
        raise NotImplementedError('Scorer "%s" is not implemented' % scorer_name)
    return perf_row_dict

  @staticmethod
  def calc_auprc(ytrue, clf, Xtest, pos_label=1, ax=None):
    if len(set(ytrue)) > 2:
      raise NotImplementedError('AUPRC is not supported for multi-class clf yet!')
    auprc = -1
    try:
      ypred_proba = clf.predict_proba(Xtest)[:, pos_label]
      precision, recall, thresholds = precision_recall_curve(ytrue, ypred_proba, pos_label=pos_label)
      auprc = auc(recall, precision)
      if ax is not None:
        plot_precision_recall_curve(clf, Xtest, ytrue, pos_label=pos_label, ax=ax, label=get_clf_name(clf))
    except Exception as e:
      print('AUROC got an exception!')
      print(e)
      warnings.warn(f"Input classifier{clf} has neither predict_proba() nor decision_function() method!")
    return auprc

  @staticmethod
  def calc_auc_roc(ytrue, clf, X, pos_label=1, ax=None):
    if pos_label == 0:
      ytrue = 1 - np.array(ytrue)
    auroc = -1
    # Multi-class classifier
    if len(set(ytrue)) > 2:
      try:
        ypred_proba = clf.predict_proba(X)
        auroc = roc_auc_score(ytrue, ypred_proba, multi_class='ovr')
      except Exception:
        warnings.warn('Multi-class classifier does not have method predict_proba(), returning -1!')
      return auroc

    # Binary classifier
    try:
      ypred_proba = clf.predict_proba(X)[:, pos_label]
      if get_clf_name(clf) == 'Ensemble':
        print('***Ensemble proba', ypred_proba)
      auroc = roc_auc_score(ytrue, ypred_proba)
      if ax is not None:
        plot_roc_curve(clf, X, ytrue, pos_label=pos_label, ax=ax, label=get_clf_name(clf))
    except Exception as e:
      print('AUROC got an exception!')
      print(e)
      warnings.warn(f"Input classifier {get_clf_name(clf)} has neither predict_proba() nor decision_function() method!")
    return auroc


# Evaluate models' performance across k trials
def eval_model_all_ktrials(k_datasets, k_model_dict, eval_by_cohort=SURG_GROUP, scorers=None,
                           eval_sda_only=False, eval_surg_only=False, years=None, care_class=None,
                           md_to_show_confmat=None, show_prc_roc=False, models=None, kt_md_dataset=None,
                           surg_agree_disagree=False, train_perf_df=None, test_perf_df=None):
  # k_datasets: {0: Dataset object, 1: Dataset object2, ...}
  # k_model_dict: {0: {'XGBoost': XGBClassifier object, 'Random Forest': RandomForestClassifier object, ...},
  #                1: {}, ...}
  _, (auroc_ax, auprc_ax) = (None, (None, None)) if not show_prc_roc else plt.subplots(1, 2, figsize=(22, 8))
  task = None
  model_to_k_confmats_test = defaultdict(dict)  # only for modeling-all, do not support cohort perf yet
  for kt, dataset_k in tqdm(k_datasets.items()):
    task = dataset_k.outcome
    # Model performance
    model_dict = k_model_dict[kt]
    if models is not None:
      model_dict = {md: model_dict[md] for md in models}
    for md, clf in model_dict.items():
      # Switch to the model-specific dataset (e.g. due to input normalizer)
      if (kt_md_dataset is not None) and (md in kt_md_dataset[kt].keys()):
        dataset = kt_md_dataset[kt][md]
      else:
        dataset = dataset_k
      # Evaluate each model on the dataset
      if eval_by_cohort is not None:
        print('Cohort-wise eval: ', md)
        train_perf_df, test_perf_df = eval_model_by_cohort(
          dataset, clf, scorers, eval_by_cohort, trial_i=kt, sda_only=eval_sda_only, surg_only=eval_surg_only,
          years=years, care_class=care_class, train_perf_df=train_perf_df, test_perf_df=test_perf_df
        )
      else:
        print('Aggergative eval: ', md)
        train_perf_df, test_perf_df, confmat_test, surg_confmat_test = eval_model(
          dataset, clf, scorers, trial_i=kt, sda_only=eval_sda_only, surg_only=eval_surg_only, years=years,
          care_class=care_class, show_confmat=md_to_show_confmat, auprc_ax=auprc_ax, auroc_ax=auroc_ax,
          surg_agree_disagree=surg_agree_disagree, train_perf_df=train_perf_df, test_perf_df=test_perf_df
        )
        model_to_k_confmats_test[md][kt] = confmat_test
        model_to_k_confmats_test[SURGEON][kt] = surg_confmat_test

  print('Models to show confusion matrix: ', model_to_k_confmats_test.keys())
  if md_to_show_confmat:
    # show confusion matrix of the median performance
    show_confmat_of_median_perf_for_mds(test_perf_df, model_to_k_confmats_test, md_to_show_confmat, Xtype='Test',
                                        criterion=SCR_ACC)
  if show_prc_roc:
    auroc_ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='No Skill')
    auroc_ax.set_title(f'Area under ROC (Task: LOS > {task[-1]})', fontsize=20, y=1.02)
    auroc_ax.set_xlabel('False positive rate', fontsize=16)
    auroc_ax.set_ylabel('True positive rate', fontsize=16)
    auroc_ax.xaxis.set_tick_params(labelsize=14)
    auroc_ax.yaxis.set_tick_params(labelsize=14)
    auroc_ax.legend()

    ht = sum(k_datasets[0].ytest) / len(k_datasets[0].ytest)  # percentage of positives
    auprc_ax.plot([0, 1], [ht, ht], linestyle='--', color='black', label='No Skill')
    auprc_ax.set_title(f'Area under PRC (Task: LOS > {task[-1]})', fontsize=20, y=1.02)
    auprc_ax.set_xlabel('Recall', fontsize=16)
    auprc_ax.set_ylabel('Precision', fontsize=16)
    auprc_ax.xaxis.set_tick_params(labelsize=14)
    auprc_ax.yaxis.set_tick_params(labelsize=14)
    auprc_ax.legend()

    plt.savefig(FIG_DIR / f'auprc_{task}-2.png', dpi=200, bbox_inches='tight')
    plt.show()

  return train_perf_df, test_perf_df


# Helper function to evaluate surgeon's performance
def eval_surgeon_perf(dataset: Dataset, scorers: Iterable, show_confmat, years=None, care_class=None,
                      selected_test_case_keys=None):
  train_scores_row_dict, test_scores_row_dict = {}, {}
  if dataset.Xtrain is not None and len(dataset.Xtrain) > 0:
    train_true_preds = dataset.get_surgeon_pred_df_by_case_key(dataset.train_case_keys, years=years, care_class=care_class)
    train_scores_row_dict = MyScorer.apply_scorers(scorers, train_true_preds[dataset.outcome],
                                                   train_true_preds[SPS_PRED], enable_warning=False)
  confmat_test = None
  if dataset.Xtest is not None and len(dataset.Xtest) > 0:
    query_test_keys = dataset.test_case_keys if selected_test_case_keys is None else np.array(selected_test_case_keys)
    test_true_preds = dataset.get_surgeon_pred_df_by_case_key(query_test_keys, years=years, care_class=care_class)
    test_scores_row_dict = MyScorer.apply_scorers(scorers, test_true_preds[dataset.outcome], test_true_preds[SPS_PRED],
                                                  enable_warning=False)
    if show_confmat:
      confmat_test = confusion_matrix(test_true_preds[dataset.outcome], test_true_preds[SPS_PRED], normalize='true')
  return train_scores_row_dict, test_scores_row_dict, confmat_test


# Helper function for eval_model_by_cohort()
def eval_model_by_cohort_Xydata(trial_i, dataset: Dataset, clf, cohort_to_XyKeys, Xtype,
                                scorers, perf_df, surg_only, years, care_class):
  # Get classifier name
  md_name = get_clf_name(clf)

  # Get year label (None means using all years in dataset)
  year_label = get_year_label(years, dataset)

  # Evaluate on each cohort, include surgeon's performance by request
  for cohort in cohort_to_XyKeys:
    X, y, cohort_case_keys = cohort_to_XyKeys.get(cohort, (np.array([]), np.array([]), np.array([])))
    if len(X) > 0:
      pred = clf.predict(X)
      scores = MyScorer.apply_scorers(scorers, y, pred, enable_warning=False)  # {'metric1': 0.82, 'metric2': 0.98, ...}
      if SCR_AUROC in scorers:
        scores[SCR_AUROC] = MyScorer.calc_auc_roc(y, clf, X)
      if SCR_AUROC_NEG in scorers:
        scores[SCR_AUROC_NEG] = MyScorer.calc_auc_roc(y, clf, X, pos_label=0)
      if SCR_AUPRC in scorers:
        scores[SCR_AUPRC] = MyScorer.calc_auprc(y, clf, X)

      class_to_counts = get_class_count(y, outcome=dataset.outcome)
      perf_df = append_perf_row_generic(
        perf_df, scores, {**class_to_counts,
                          **{'Xtype': Xtype, 'Cohort': cohort, 'Model': md_name, 'Count': X.shape[0],
                              'Trial': trial_i, 'Year': year_label}
                          })
      if surg_only:
        true_surg_preds = dataset.get_surgeon_pred_df_by_case_key(cohort_case_keys, years=years, care_class=care_class)
        scores_surg = MyScorer.apply_scorers(scorers, true_surg_preds[dataset.outcome], true_surg_preds[SPS_PRED],
                                             enable_warning=False)
        perf_df = append_perf_row_generic(
          perf_df, scores_surg, {**class_to_counts,
                                 **{'Xtype': Xtype, 'Cohort': cohort, 'Model': SURGEON, 'Count': X.shape[0],
                                    'Trial': trial_i, 'Year': year_label}
                                 })
  return perf_df


# Evaluates the model performance on each cohort
def eval_model_by_cohort(dataset: Dataset, clf, scorers=None, cohort_type=SURG_GROUP, trial_i=None,
                         sda_only=False, surg_only=False, years=None, care_class=None,
                         train_perf_df=None, test_perf_df=None):
  assert cohort_type in COHORT_TYPE_SET, f'cohort_type must be ont of {COHORT_TYPE_SET}'

  if scorers is None:
    scorers = deepcopy(DEFAULT_SCORERS)
  if train_perf_df is None:
    train_perf_df = get_default_perf_df(scorers)
  if test_perf_df is None:
    test_perf_df = get_default_perf_df(scorers)
  print('Scorers: ', scorers)

  # Evaluate on Training set for each cohort: {'cohort1': (X_cohort1, y_cohort1), '': (), }
  cohort_to_XyKeys_train = dataset.get_cohort_to_Xytrains(cohort_type, sda_only=sda_only, surg_only=surg_only, years=years)
  train_perf_df = eval_model_by_cohort_Xydata(trial_i, dataset, clf, cohort_to_XyKeys_train, 'train', scorers=scorers,
                                              perf_df=train_perf_df, surg_only=surg_only, years=years, care_class=care_class)

  # Evaluate on Test set for each cohort
  cohort_to_XyKeys_test = dataset.get_cohort_to_Xytests(cohort_type, sda_only=sda_only, surg_only=surg_only, years=years)
  test_perf_df = eval_model_by_cohort_Xydata(trial_i, dataset, clf, cohort_to_XyKeys_test, 'test', scorers=scorers,
                                             perf_df=test_perf_df, surg_only=surg_only, years=years, care_class=care_class)

  train_perf_df.sort_values(by=['Count', 'Cohort', 'Model'], ascending=False, inplace=True)  # 'accuracy'
  test_perf_df.sort_values(by=['Count', 'Cohort', 'Model'], ascending=False, inplace=True)  # 'accuracy'
  return train_perf_df, test_perf_df


# Evaluate model on various groups of cases (e.g. pure SDA cases, cases with surgeon prediction, all cases etc.)
def eval_model(dataset: Dataset, clf, scorers=None, trial_i=None, sda_only=False, surg_only=False, years=None,
               cohort='All', care_class=None, show_confmat=False, auprc_ax=None, auroc_ax=None,
               surg_agree_disagree=False, train_perf_df=None, test_perf_df=None):
  if scorers is None:
    scorers = deepcopy(DEFAULT_SCORERS)
  if train_perf_df is None:
    train_perf_df = get_default_perf_df(scorers)
  if test_perf_df is None:
    test_perf_df = get_default_perf_df(scorers)
  print('Scorers: ', scorers)
  # Get classifier name
  md_name = get_clf_name(clf)

  # Get outcome name
  outcome = dataset.outcome

  # Get year label (None means using all years in dataset)
  year_label = get_year_label(years, dataset)

  # Get train & test X, y under sda, surg, years filters
  Xtrain, ytrain = dataset.get_Xytrain_by_case_key(dataset.train_case_keys, care_class=care_class,
                                                   sda_only=sda_only, surg_only=surg_only, years=years)
  Xtest, ytest, test_case_keys = dataset.get_Xytest_by_case_key(dataset.test_case_keys, care_class=care_class,
                                                                sda_only=sda_only, surg_only=surg_only, years=years)
  # Apply trained clf and evaluate
  if Xtrain is not None and len(Xtrain) > 0:
    train_pred = clf.predict(Xtrain)
    train_scores = MyScorer.apply_scorers(scorers, ytrain, train_pred, enable_warning=False)
    if SCR_AUROC in scorers:
      train_scores[SCR_AUROC] = MyScorer.calc_auc_roc(ytrain, clf, Xtrain)
    if SCR_AUROC_NEG in scorers:
      train_scores[SCR_AUROC_NEG] = MyScorer.calc_auc_roc(ytrain, clf, Xtrain, pos_label=0)
    if SCR_AUPRC in scorers:
      train_scores[SCR_AUPRC] = MyScorer.calc_auprc(ytrain, clf, Xtrain)
    train_perf_df = append_perf_row_generic(
      train_perf_df, train_scores, {**get_class_count(ytrain, outcome=outcome),
                                    **{'Xtype': 'train', 'Cohort': cohort, 'Model': md_name, 'Trial': trial_i,
                                       'Count': Xtrain.shape[0], 'Year': year_label}})
  confmat_test = None
  select_idx = np.arange(len(Xtest))
  selected_test_case_keys = None
  if Xtest is not None and len(Xtest) > 0:
    # get model prediction
    test_pred = clf.predict(Xtest)

    # get surgeon's prediction
    if surg_only:
      tmp = dataset.get_surgeon_pred_df_by_case_key(dataset.test_case_keys, years=years, care_class=care_class).set_index('SURG_CASE_KEY')
      surg_pred_test = np.array(tmp.loc[test_case_keys, SPS_PRED])
    # print(tmp.shape, dataset.test_case_keys.shape, ytest.shape, tmp.head())
    # print(f'{md_name} surg_ytest == ytest: ', np.all(surg_pred_ytest == ytest))
    # print(f'{md_name} surg_ytest_case_keys set == ytest_case_keys set: ', set(tmp.index.to_numpy()) == set(test_case_keys))
    # print(f'{md_name} surg_ytest_case_keys == ytest_case_keys: ', np.all(tmp.index.to_numpy()) == test_case_keys)
      if surg_agree_disagree and (len(surg_pred_test) != len(test_pred)):
        warnings.warn('Surgeon Agreement Check is only available on cases with SPS prediction!')
      if surg_agree_disagree == XAGREE:
        select_idx = np.where(test_pred == surg_pred_test)[0]
      elif surg_agree_disagree == XDISAGREE:
        select_idx = np.where(test_pred != surg_pred_test)[0]
      elif surg_agree_disagree == XDISAGREE1:
        select_idx = np.where(np.abs(surg_pred_test - test_pred) == 1)[0]
      elif surg_agree_disagree == XDISAGREE_GT1:
        select_idx = np.where(np.abs(surg_pred_test - test_pred) > 1)[0]
      Xtest, selected_test_case_keys = Xtest[select_idx, :], test_case_keys[select_idx]
      ytest, test_pred = ytest[select_idx], test_pred[select_idx]

    # Apply scorers to evaluate prediction performance
    test_scores = MyScorer.apply_scorers(scorers, ytest, test_pred, enable_warning=False)
    if SCR_AUROC in scorers:
      test_scores[SCR_AUROC] = MyScorer.calc_auc_roc(ytest, clf, Xtest, ax=auroc_ax)
    if SCR_AUROC_NEG in scorers:
      test_scores[SCR_AUROC_NEG] = MyScorer.calc_auc_roc(ytest, clf, Xtest, pos_label=0)
    if SCR_AUPRC in scorers:
      test_scores[SCR_AUPRC] = MyScorer.calc_auprc(ytest, clf, Xtest, ax=auprc_ax)
    # Save to performance result dataframe
    test_perf_df = append_perf_row_generic(
      test_perf_df, test_scores, {**get_class_count(ytest, outcome=outcome),
                                  **{'Xtype': 'test', 'Cohort': cohort, 'Model': md_name, 'Trial': trial_i,
                                     'Count': Xtest.shape[0], 'Year': year_label}})
    if show_confmat:
      confmat_test = confusion_matrix(ytest, test_pred, normalize='true')

  # Surgeon performance
  surg_confmat_test = None
  if surg_only:
    surg_train, surg_test, surg_confmat_test = eval_surgeon_perf(dataset, scorers, show_confmat,
                                                                 years=years,
                                                                 care_class=care_class,
                                                                 selected_test_case_keys=selected_test_case_keys)
    if len(surg_train) > 0:
      train_perf_df = append_perf_row_generic(
        train_perf_df, surg_train, {**get_class_count(ytrain, outcome=dataset.outcome),
                                    **{'Xtype': 'train', 'Cohort': cohort, 'Model': SURGEON, 'Trial': trial_i,
                                       'Count': Xtrain.shape[0], 'Year': year_label}})
    if len(surg_test) > 0:
      surg_model = SURGEON if surg_agree_disagree in {XAGREE, None, False} else f'{SURGEON}-{md_name}-disagree'
      test_perf_df = append_perf_row_generic(
        test_perf_df, surg_test, {**get_class_count(ytest, outcome=dataset.outcome),
                                  **{'Xtype': 'test', 'Cohort': cohort, 'Model': surg_model,
                                     'Trial': trial_i, 'Count': Xtest.shape[0], 'Year': year_label}})
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



# def gen_shapley_values(dataset: Dataset, clf, is_ensemble=False):
#   if is_ensemble:
#     shap_vals_treeMds = []
#     for md in [LGR, DTCLF, RMFCLF, EXTREECLF, XGBCLF]:
#       # for md in [EXTREECLF]:
#       if md == LGR:
#         explainer = shap.Explainer(clf.base_estimator, full_dataset_std_POSTOP.Xtrain)
#       else:
#         explainer = shap.TreeExplainer(clf)
#       shap_vals = explainer.shap_values(os_dataset_std_POSTOP.Xtest)
#       shap_vals_treeMds.append(shap_vals)
#
#   pass


@DeprecationWarning
def gen_feature_importance(model, mdabbr, reg=True, ftrs=FEATURE_COLS, pretty_print=False, plot_top_K=None):
  # ftrs must match the data matrix features exactly (same order)
  sorted_frts = [(ftr, imp) for ftr, imp in sorted(zip(ftrs, model.feature_importances_), reverse=True, key=lambda i: i[1])]
  # sorted_frts = [(x, y) for y, x in sorted(zip(model.feature_importances_, ftrs), reverse=True, key=lambda p: p[0])]
  if pretty_print:
    print("\n" + reg2name[mdabbr] if reg else clf2name[mdabbr] + ":")
    c = 1
    for x, y in sorted_frts:
      print("{c}.{ftr}:  {score}".format(c=c, ftr=x, score=round(y, 4)))
      c += 1
  if plot_top_K is not None:
    ftr_importance = sorted_frts[:plot_top_K]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([x[0] for x in ftr_importance], [x[1] for x in ftr_importance])
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Features")
    ax.set_title("Top %d most important features (%s)" % (plot_top_K, mdabbr))
    plt.show()

  # shap.summary_plot()
  return sorted_frts
