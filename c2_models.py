# Customized Model classes
import warnings

import numpy as np
import optuna
from collections import Counter
from tqdm import tqdm
from typing import Any, Dict, List

from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PoissonRegressor, Ridge, RidgeCV
from sklearn.naive_bayes import CategoricalNB, ComplementNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
# from mlens.ensemble import SuperLearner

from globals import *
from c1_data_preprocessing import Dataset
from c4_model_perf import MyScorer


try:
  from statsmodels.miscmodels.ordinal_model import OrderedModel
  # An extensible wrapper classifier of statsmodels's OrderedModel()
  class OrdinalClassifier(object):

    def __init__(self, distr='logit', solver='lbgfs', disp=False, maxiter=200):
      self.distr = distr
      self.solver = solver
      self.disp = disp
      self.maxiter = maxiter
      self.ord_model = None
      self.fitted_ord_model = None

    def fit(self, X, y):
      self.ord_model = OrderedModel(endog=y, exog=X, distr=self.distr)
      self.ord_fitted_model = self.ord_model.fit(method=self.solver, maxiter=self.maxiter, disp=self.disp)
      return self

    def predict(self, X):
      proba = self.predict_proba(X)
      return np.argmax(proba, axis=1)

    def predict_proba(self, X):
      return self.ord_fitted_model.model.predict(params=self.ord_fitted_model.params, exog=X, which='prob')
except ModuleNotFoundError as e:
  print(e)
  print('OrderedClassifier() is not defined!')
except ImportError as e:
  print(e)
  print('OrderedClassifier() is not defined!')


# Abstract class for cross validation
class ClassifierCV(object):

  def __init__(self):
    self.clf = None

  def fit(self, X, y):
    self.clf.fit(X, y)

  def predict(self, X):
    pred = self.clf.predict(X)
    pred[pred > MAX_NNT] = MAX_NNT + 1
    return pred

  def score(self, X, y, sample_weight=None):
    preds = self.predict(X)
    return accuracy_score(y, preds, sample_weight=sample_weight)


# Regression-based classifier (regressor + round to nearest int)
class RegressionBasedClassifier(object):

  def __init__(self, md, **kwargs):
    self.regressor = None
    if md == PR:
      self.regressor = PoissonRegressor(**kwargs)  # alpha=0.01, max_iter=300
    elif md == SVR:
      self.regressor = SVR(**kwargs)
    elif md == KNR:
      self.regressor = KNeighborsRegressor(**kwargs)
    elif md == DT:
      self.regressor = DecisionTreeRegressor(**kwargs)
    elif md == RMF:
       self.regressor = RandomForestRegressor(**kwargs)
    elif md == GB:
      self.regressor = GradientBoostingRegressor(**kwargs)
    elif md == XGB:
      self.regressor = XGBRegressor(**kwargs)
    elif md == MLP:
      self.regressor = MLPRegressor(**kwargs)
    else:
      raise NotImplementedError

  def fit(self, X, y):
    self.regressor.fit(X, y)

  def predict(self, X):
    reg_pred = np.rint(self.regressor.predict(X))
    reg_pred[reg_pred > MAX_NNT] = MAX_NNT + 1
    return reg_pred

  def score(self, X, y, sample_weight=None):
    preds = self.predict(X)
    return accuracy_score(y, preds, sample_weight=sample_weight)


# Wrapper for all model objects for safe handling a single class in training data
class SafeOneClassWrapper(BaseEstimator, ClassifierMixin):
  def __init__(self, base_estimator):
    self.base_estimator = base_estimator
    self.model_type = base_estimator.__class__.__name__

  def fit(self, X, y, **kwargs):
    try:
      return self.base_estimator.fit(X, y, **kwargs)
    except ValueError as exc:
      if not str(exc).startswith('This solver needs samples of at least 2 classes in the data'):
        raise
    finally:
      self.classes_ = np.unique(y)

  def decision_function(self, X):
    if len(self.classes_) == 1:
      return np.ones((X.shape[1], 1))
    return self.base_estimator.decision_function(X)

  def predict_proba(self, X):
    if len(self.classes_) == 1:
      return np.ones((X.shape[1], 1))
    return self.base_estimator.predict_proba(X)

  def predict(self, X):
    if len(self.classes_) == 1:
      return np.full(X.shape[0], self.classes_[0])
    return self.base_estimator.predict(X)

  def score(self, X, y, sample_weight=None):
    ypred = self.predict(X)
    return accuracy_score(y, ypred, sample_weight=sample_weight)


# CV wrapper for KNN TODO: add other searchCV methods!
class KNeighborsClassifierCV(ClassifierCV):

  def __init__(self, param_space, kfold=5, searchCV=GRID_SEARCH, scoring=SCR_ACC):
    super().__init__()
    if searchCV == GRID_SEARCH:
      self.clf = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_space, scoring=scoring, cv=kfold,
                              n_jobs=-1, refit=True, return_train_score=True, verbose=0)
    elif searchCV == OPTUNA_SEARCH:
      self.clf = optuna.integration.OptunaSearchCV(estimator=KNeighborsClassifier, scoring=scoring, cv=kfold, n_jobs=-1,
                                                   refit=True)
    else:
      raise NotImplementedError
