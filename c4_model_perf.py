import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import Dict, Any

from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer, roc_auc_score, mean_squared_error
# from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge, RidgeCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC, SVR
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
# from xgboost import XGBClassifier
import shap

import globals
from c1_data_preprocessing import Dataset


class MyScorer:

  def __init__(self):
    return

  @staticmethod
  def get_scorer_dict(scorer_names):
    scr_dict = {}
    for scorer in scorer_names:
      if scorer == globals.SCR_ACC:
        scr_dict[scorer] = 'accuracy'
      elif scorer == globals.SCR_ACC_BAL:
        scr_dict[scorer] = 'balanced_accuracy'
      elif scorer == globals.SCR_AUC:
        scr_dict[scorer] = 'roc_auc'
      elif scorer == globals.SCR_RMSE:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_rmse, greater_is_better=False)
      elif scorer == globals.SCR_ACC_ERR1:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_1nnt_tol, greater_is_better=True)
      elif scorer == globals.SCR_ACC_ERR2:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_2nnt_tol, greater_is_better=True)
      elif scorer == globals.SCR_OVERPRED:
        scr_dict[scorer] = make_scorer(MyScorer.scorer_overpred_pct, greater_is_better=False)
      elif scorer == globals.SCR_UNDERPRED:
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
      if scorer_name == globals.SCR_ACC:
        perf_row_dict[scorer_name] = accuracy_score(ytrue, ypred)
      elif scorer_name == globals.SCR_ACC_BAL:
        perf_row_dict[scorer_name] = balanced_accuracy_score(ytrue, ypred)
      elif scorer_name == globals.SCR_RMSE:
        perf_row_dict[scorer_name] = MyScorer.scorer_rmse(ytrue, ypred)
      elif scorer_name == globals.SCR_ACC_ERR1:
        perf_row_dict[scorer_name] = MyScorer.scorer_1nnt_tol(ytrue, ypred)
      elif scorer_name == globals.SCR_ACC_ERR2:
        perf_row_dict[scorer_name] = MyScorer.scorer_2nnt_tol(ytrue, ypred)
      elif scorer_name == globals.SCR_OVERPRED:
        perf_row_dict[scorer_name] = MyScorer.scorer_overpred_pct(ytrue, ypred)
      elif scorer_name == globals.SCR_UNDERPRED:
        perf_row_dict[scorer_name] = MyScorer.scorer_underpred_pct(ytrue, ypred)
      elif scorer_name == globals.SCR_AUC:
        perf_row_dict[scorer_name] = None  # TODO: find a better solution to handle this
        # perf_row_dict[scorer_name] = roc_auc_score(ytrue, ypred)  # ypred is actually yscore: clf.predict_proba(X)[:, 1]
      else:
        raise NotImplementedError('%s not implemented' % scorer_name)
    return perf_row_dict


def append_perf_row_generic(perf_df, score_dict: Dict, add_col_dict: Dict[str, Any]):
  score_dict.update(add_col_dict)
  perf_df.append(score_dict, ignore_index=True)
  return perf_df


def append_perf_row(perf_df, trial, md, scores_row_dict: Dict):
  scores_row_dict['Model'] = md
  scores_row_dict['Trial'] = trial
  perf_df.append(scores_row_dict, ignore_index=True)
  return perf_df


def append_perf_row_surg(surg_perf_df: pd.DataFrame, trial, scores_row_dict):
  if scores_row_dict is None or len(scores_row_dict) == 0:
    return surg_perf_df
  scores_row_dict['Trial'] = trial
  scores_row_dict['Model'] = 'Surgeon-train'
  surg_perf_df.append(scores_row_dict, ignore_index=True)
  return surg_perf_df


def eval_surgeon_perf(dataset: Dataset, scorers):
  train_scores_row_dict, test_scores_row_dict = {}, {}
  if dataset.Xtrain:
    train_surg_pred = dataset.get_surgeon_pred_df_by_case_key(dataset.train_case_keys)[globals.SPS_PRED]
    train_scores_row_dict = MyScorer.apply_scorers(scorers, dataset.ytrain, train_surg_pred)
  if dataset.Xtest:
    test_surg_pred = dataset.get_surgeon_pred_df_by_case_key(dataset.test_case_keys)[globals.SPS_PRED]
    test_scores_row_dict = MyScorer.apply_scorers(scorers, dataset.ytest, test_surg_pred)
  return train_scores_row_dict, test_scores_row_dict


# if __name__ == '__main__':
#   ytrue = np.ones(10)
#   ypred = np.array([0, 1] * 5)
#   scorers = MyScorer.get_scorer_dict([globals.SCR_ACC, globals.SCR_ACC_BAL, globals.SCR_ACC_ERR1, globals.SCR_ACC_ERR2,
#                                       globals.SCR_OVERPRED, globals.SCR_UNDERPRED, globals.SCR_RMSE])
#   perf_row_dict = MyScorer.apply_scorers(scorers, ytrue, ypred)
#
#   print(perf_row_dict)


# def init_perf_df(md, scorers):
#   # acc, acc_1nnt_tol, overpred_rate, underpred_rate
#   md2Classifier = {globals.LGR: LogisticRegression(), globals.SVC: SVC(), globals.KNN: KNeighborsClassifier(),
#                    globals.DTCLF: DecisionTreeClassifier(), globals.RMFCLF: RandomForestClassifier(),
#                    globals.GBCLF: GradientBoostingClassifier(), globals.XGBCLF: XGBClassifier()}
#   params = md2Classifier.get(md, [])
#   perf_df = pd.DataFrame(columns=params)
#   # top K most important features [optional, since some models do not have this]
#   return
#
#
# def add_perf_row(perf_df, new_entry):
#
#   return perf_df
#
#
# def make_perf_row(clf, X, y, fold):
#
#   return
