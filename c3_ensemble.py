"""
Definitions of ensemble models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, AnyStr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from globals import *
from c2_models import *
from c2_models_nnt import *
from utils_eval import get_clf_name


def get_default_voter_md_abbr():
  return [LGR, KNN, GAUSSNB, ADABOOST, BAGCLF, EXTREECLF, DTCLF, RMFCLF, XGBCLF]


def make_votingClf_single_bin(model_dict, base_models=None, voting='hard'):
  # if base_models is None, use all models in the k_model_dict
  base_model_dict = model_dict if base_models is None else {k: v for k, v in model_dict.items() if k in base_models}
  base_models = list(base_model_dict.items())
  voting_clf = VotingClassifier(estimators=base_models, voting=voting)
  return {'VotingClf': voting_clf}


def make_votingClf_all_bins(bin_model_dict, base_models=None, voting='hard'):
  bin_voting_clf = {}
  for bn, model_dict in bin_model_dict.items():
    bin_voting_clf[bn] = {0: make_votingClf_single_bin(model_dict, base_models, voting)}
  return bin_voting_clf


def add_voting_ensemble_binclf(bin_k_model_dict, base_models=None, skip_models=[]):
  new_bin_k_model_dict = {}
  for bn, k_model_dict in bin_k_model_dict.items():
    new_bin_k_model_dict[bn] = add_voting_ensemble(k_model_dict, base_models, skip_models)
  return new_bin_k_model_dict


def add_voting_ensemble(k_model_dict, base_models=None, skip_models=[]):
  new_k_model_dict = {}
  for k, model_dict in k_model_dict.items():
    if base_models is None:
      voting_clf = Ensemble(tasks=[TASK_MULTI_CLF],
                            md2clfs={md: {TASK_MULTI_CLF: [clf]} for md, clf in model_dict.items() if md not in skip_models})
    else:
      voting_clf = Ensemble(tasks=[TASK_MULTI_CLF],
                            md2clfs={md: {TASK_MULTI_CLF: [clf]} for md, clf in model_dict.items() if md in base_models},)
    new_k_model_dict[k] = {**model_dict, **{ENSEMBLE_MAJ_EQ: voting_clf}}
  return new_k_model_dict


# def add_and_fit_super_learner(k_model_dict, base_models=None, skip_models=[]):
#   new_k_model_dict = {}
#   for k, model_dict in k_model_dict.items():
#     if base_models is None:
#       voting_clf = SuperLearner(base_estimators=list(model_dict.values()))
#     else:
#       voting_clf = Ensemble(tasks=[TASK_MULTI_CLF],
#                             md2clfs={md: {TASK_MULTI_CLF: [clf]} for md, clf in model_dict.items() if
#                                      md in base_models}, )
#     new_k_model_dict[k] = {**model_dict, **{ENSEMBLE_MAJ_EQ: voting_clf}}
#   return new_k_model_dict


# TODO: In the future, need to allow other weighting schemes, e.g. by predict_proba, --> can create a separate class
class Ensemble(object):
  """
  Assume input classifiers are already trained
  """

  def __init__(self, tasks: List, md2clfs: Dict[AnyStr, Dict[AnyStr, List]], md_weights=None):
    self.tasks = tasks
    self.md2clfs = md2clfs  # {md: {task1: [clf1, clf2, ...], }, }
    self.md_weights = md_weights if md_weights is not None else {md: np.ones_like(tasks) for md in md2clfs.keys()}
    self.md2taskpreds = None
    self._votes_over_nnt = None
    self.check_args()

  def check_args(self):
    assert len(self.tasks) > 0, "Please specify a list of tasks"
    assert len(self.md2clfs) > 0, "Please specify a dict of model abbr to trained classifier"
    for tsk in self.tasks:
      if tsk not in ALL_TASKS:
        raise NotImplementedError("Task %s is not supported yet!" % tsk)
      for md, tsk2clfs in self.md2clfs.items():
        # if md not in ALL_MODELS:
        #   raise NotImplementedError("Model %s is not supported yet!" % md)
        if len(tsk2clfs) == 0:
          raise ValueError("Model %s is not trained on any task!" % md)

  def predict(self, X):
    # TODO: assign md2taskpreds here!! Do NOT init in __init__
    N = X.shape[0]
    md2taskpreds = {md: {tsk: [] for tsk in self.tasks} for md in self.md2clfs.keys()}
    for task in self.tasks:
      if task == MULTI_CLF:
        for md, task2clfs in self.md2clfs.items():
          print("Predicting with multiclf: %s" % md)
          md2taskpreds[md][task].append(task2clfs[task][0].predict(X))

      elif task == BIN_CLF:
        for md, task2clfs in self.md2clfs.items():
          if len(task2clfs[task]) != len(NNT_CUTOFFS):
            print("Skip '%s' that does not have binary classfication" % md)
            continue
          print("Predicting with binclf: %s" % md)
          for nnt in NNT_CUTOFFS:
            pred = task2clfs[task][nnt].predict(X)
            md2taskpreds[md][task].append(pred)
    self.md2taskpreds = md2taskpreds

    # Ensemble Rule 1: Equal weights for each model
    self._votes_over_nnt = self._get_vote_counts(N, NNT_CLASS_CNT, how='uniform')

    # Predict based on the counted votes for each class; Majority wins
    final_preds = self._predict_via_counted_votes()

    # TODO: save probability somewhere for future extension
    # TODO: analyze the proportion of tie-breaking
    # TODO: other ensemble rules to try -- e.g. 1 * multi-clf pred + proba * bin
    return final_preds

  def predict_proba(self, X):
    """
    Outputs the probability of each class being the true outcome.
    This takes the average of the class probability of all its base estimators.
    """
    self.predict(X)
    base_estimators = {md: self.md2clfs[md][MULTI_CLF][0] for md in self.md2clfs}
    base_clf_probas = np.asarray([clf.predict_proba(X) for clf in base_estimators.values()])
    avg_proba = np.average(base_clf_probas, axis=0,)
    return avg_proba

  def get_ensemble_upper_bound(self, ytrue):
    # calculate the maximum possible accuracy: as long as the true class has at least 1 vote
    if self._votes_over_nnt is None:
      raise ValueError("This ensemble model hasn't counted votes yet!")

    N = self._votes_over_nnt.shape[0]
    true_nnt_votes = self._votes_over_nnt[np.arange(N), ytrue]
    total_hits = np.count_nonzero(true_nnt_votes)
    return total_hits / N

  def _get_vote_counts(self, n_samples, n_classes, how='uniform'):
    votes_over_nnt = np.zeros((n_samples, n_classes))
    if how == 'uniform':
      for md, task2preds in self.md2taskpreds.items():
        for task, preds in task2preds.items():
          if task == MULTI_CLF:
            votes_over_nnt[np.arange(n_samples), preds[0].astype(int)] += 1
          elif task == BIN_CLF:
            # If pred == 1 at cutoff=c, add vote count by 1 for all classes<=c
            for le_nnt in range(len(preds)):
              votes_over_nnt[preds[le_nnt] == 1, :le_nnt+1] += 1
              votes_over_nnt[preds[le_nnt] == 0, le_nnt+1:] += 1
    else:
      raise NotImplementedError("Ensemble rule '%s' is not implemented yet!" % how)

    return votes_over_nnt

  def _predict_via_counted_votes(self):
    majority_class = np.argmax(self._votes_over_nnt, axis=1)
    return majority_class

  def get_votes_std(self, ytrue=None):
    # generate the distribution of the votes for each class
    if self._votes_over_nnt is None:
      raise ValueError("This ensemble model hasn't counted votes yet!")
    counts = np.sum(self._votes_over_nnt, axis=1)
    bins = np.array(NNT_CLASSES)
    mean = np.sum(self._votes_over_nnt * bins, axis=1) / counts

    sum_squares = np.sum(bins**2 * self._votes_over_nnt, axis=1)
    var = sum_squares / counts - mean**2
    return np.sqrt(var)

  def score(self, X, y):
    # Accuracy
    ypred = self.predict(X)
    acc = accuracy_score(y, ypred)
    return acc

  def __str__(self):
    s = 'Ensemble Model\n' + str(self.md2clfs)
    return s


def get_default_base_models(is_binary=False):
  models = {}
  # models.append(make_pipeline(StandardScaler(), GaussianNB()))
  # models.append(SVC(gamma='scale', probability=True))
  models[LGR] = get_model(LGR, cls_weight=None) if not is_binary else get_model_binclf(LGR, cls_weight=None)
  models[SVC_LINEAR] = get_model(SVC_LINEAR, cls_weight=None) if not is_binary else get_model_binclf(SVC_LINEAR, cls_weight=None)
  models[KNN] = get_model(KNN, cls_weight=None) if not is_binary else get_model_binclf(KNN, cls_weight=None)
  models[ADABOOST] = AdaBoostClassifier(n_estimators=100, random_state=SEED)
  models[EXTREECLF] = get_model(EXTREECLF, cls_weight=None) if not is_binary else get_model_binclf(EXTREECLF, cls_weight=None)
  models[BAGCLF] = get_model(BAGCLF) if not is_binary else get_model_binclf(BAGCLF, cls_weight=None)
  models[DTCLF] = get_model(DTCLF, cls_weight=None) if not is_binary else get_model_binclf(DTCLF, cls_weight=None)
  models[RMFCLF] = get_model(RMFCLF, cls_weight=None) if not is_binary else get_model_binclf(RMFCLF, cls_weight=None)
  models[XGBCLF] = get_model(XGBCLF, cls_weight=None) if not is_binary else get_model_binclf(XGBCLF, cls_weight=None)
  return models


class SuperLearner(object):
  # TODO: note that KNN cannot be one of the base estimators, since it doesn't have predict_proba()
  def __init__(self, base_estimators, meta_estimator_type=LGR, base_fitted=False):
    self.base_models = base_estimators
    self.base_fitted = base_fitted
    if meta_estimator_type == LGR:
      self.meta_model = LogisticRegression(solver='liblinear')
    else:
      raise NotImplementedError
    #self.meta_X, self.meta_y = None, None
    # TODO: pass md_to_scaler dict in init()
    # TODO: pass numeric_idx & input scaler_dict to fit() & predict()

  def fit(self, X, y, kfold=10):
    meta_X, meta_y = self._get_out_of_fold_predictions(X, y, kfd=kfold)
    self._fit_base_model(X, y)
    self._fit_meta_model(meta_X, meta_y)

  def predict(self, X):
    # if not self.base_fitted:
    #   raise Exception('Base_estimators are not fitted yet!')
    # build meta_X
    meta_X = list()
    for md, clf in self.base_models.items():
      yhat = clf.predict_proba(X)
      meta_X.append(yhat)
    meta_X = np.hstack(meta_X)
    # predict via meta model
    return self.meta_model.predict(meta_X)

  def _get_out_of_fold_predictions(self, X, y, kfd):
    meta_X, meta_y = list(), list()
    # define split of data
    kfold = KFold(n_splits=kfd, shuffle=True, random_state=SEED)
    # enumerate splits
    for train_idx, test_idx in tqdm(kfold.split(X), total=kfd):
      fold_yhats = list()
      # get data
      train_X, test_X = X[train_idx], X[test_idx]
      train_y, test_y = y[train_idx], y[test_idx]
      meta_y.extend(test_y)
      # fit and make predictions with each sub-model
      for md, clf in self.base_models.items():
        print(f'[super-learner] {md}')
        clf.fit(train_X, train_y)
        yhat = clf.predict_proba(test_X)  # --> (n_samples, n_classes)
        # store columns
        fold_yhats.append(yhat)
      # store fold yhats as columns
      meta_X.append(np.hstack(fold_yhats))
    # meta_X: (N_samples, n_classes * n_base_models)
    return np.vstack(meta_X), np.asarray(meta_y)

  def _fit_base_model(self, X, y):
    if not self.base_fitted:
      for clf in self.base_models.values():
        clf.fit(X, y)
      self.base_fitted = True

  def _fit_meta_model(self, X, y):
    self.meta_model.fit(X, y)


class SuperLearnerV2(BaseEstimator):

  def __init__(self, sl_v1: SuperLearner):
    self.sl_v1 = sl_v1

  def predict_proba(self, X):
    if not self.sl_v1.base_fitted:
      raise Exception('Base_estimators are not fitted yet!')
    # build meta_X
    meta_X = list()
    for md, clf in self.sl_v1.base_models.items():
      yhat = clf.predict_proba(X)
      meta_X.append(yhat)
    meta_X = np.hstack(meta_X)
    proba = self.sl_v1.meta_model.predict_proba(meta_X)
    print(proba)
    return proba

  def predict(self, X):
    return self.sl_v1.predict(X)

  def fit(self, X, y, kfold=10):
    self.sl_v1.fit(X, y, kfold)
