"""
Definitions of ensemble models
"""
from typing import Dict, List, AnyStr
import numpy as np
from sklearn.metrics import accuracy_score

import los_prediction_global_vars as globals


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
      if tsk not in globals.ALL_TASKS:
        raise NotImplementedError("Task %s is not supported yet!" % tsk)
      for md, tsk2clfs in self.md2clfs.items():
        if md not in globals.ALL_CLFS:
          raise NotImplementedError("Model %s is not supported yet!" % md)
        if len(tsk2clfs) == 0:
          raise ValueError("Model %s is not trained on any task!" % md)
        # if tsk2clfs.get(tsk, None) is None or len(tsk2clfs[tsk]) == 0:
        #   raise ValueError("Model %s is not trained on task %s" % (md, tsk))
        # try:
        #   check_is_fitted()

  def predict(self, X):
    N = X.shape[0]
    md2taskpreds = {md: {tsk: [] for tsk in self.tasks} for md in self.md2clfs.keys()}
    for task in self.tasks:
      if task == globals.TASK_MULTI_CLF:
        for md, task2clfs in self.md2clfs.items():
          print("Predicting with multiclf: %s" % md)
          md2taskpreds[md][task].append(task2clfs[task][0].predict(X))

      elif task == globals.TASK_BIN_CLF:
        for md, task2clfs in self.md2clfs.items():
          if len(task2clfs[task]) != len(globals.NNT_CUTOFFS):
            print("Skip '%s' that does not have binary classfication" % md)
            continue
          print("Predicting with binclf: %s" % md)
          for nnt in globals.NNT_CUTOFFS:
            pred = task2clfs[task][nnt].predict(X)
            md2taskpreds[md][task].append(pred)
    self.md2taskpreds = md2taskpreds

    # Ensemble Rule 1: Equal weights for each model
    self._votes_over_nnt = self._get_vote_counts(N, globals.NNT_CLASS_CNT, how='uniform')

    # Predict based on the counted votes for each class; Majority wins
    final_preds = self._predict_via_counted_votes()

    return final_preds

  def predict_proba(self, X):
    """
    Outputs the probability of each class being the true outcome.
    This is identical to the proportion of the votes in each class.
    """
    self.predict(X)
    rowsum = np.sum(self._votes_over_nnt, axis=1)
    return self._votes_over_nnt / rowsum

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
          if task == globals.TASK_MULTI_CLF:
            votes_over_nnt[np.arange(n_samples), preds[0].astype(int)] += 1
          elif task == globals.TASK_BIN_CLF:
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

  def score(self, X, y):
    # Accuracy
    ypred = self.predict(X)
    acc = accuracy_score(y, ypred)
    return acc

  def __str__(self):
    s = 'Ensemble Model\n' + str(self.md2clfs)
    return s

