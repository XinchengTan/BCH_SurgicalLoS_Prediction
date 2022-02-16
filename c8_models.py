# Customized Model classes
import json
import numpy as np
from collections import Counter

import xgboost
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PoissonRegressor, Ridge, RidgeCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier

from . import globals
from .c5_model_perf import MyScorer


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


class PoissonClassifier(PoissonRegressor):

  def __init__(self, alpha=1, fit_intercept=True, max_iter=100, tol=1e-4):
    super().__init__(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)

  def predict(self, X):
    reg_pred = np.rint(super().predict(X))
    reg_pred[reg_pred > globals.MAX_NNT] = globals.MAX_NNT + 1
    return reg_pred

  def score(self, X, y, sample_weight=None):
    preds = self.predict(X)
    return accuracy_score(y, preds, sample_weight=sample_weight)


def train_model(md, params, X, y):
  if md == globals.LGR:
    clf = LogisticRegression(random_state=globals.SEED)
  elif md == globals.SVC:
    clf = SVC(random_state=globals.SEED, probability=True)
  elif md == globals.KNN:
    clf = KNeighborsClassifier()
  elif md == globals.DTCLF:
    clf = DecisionTreeClassifier(random_state=globals.SEED)
  elif md == globals.RMFCLF:
    clf = RandomForestClassifier(random_state=globals.SEED)
  elif md == globals.GBCLF:
    clf = GradientBoostingClassifier(random_state=globals.SEED)
  elif md == globals.XGBCLF:
    clf = XGBClassifier(random_state=globals.SEED)
  else:
    raise NotImplementedError(f"Model {md} is not implemented yet!")

  clf.set_params(**params)
  clf.fit(X, y)
  return clf


def train_model_cv(md, X, y, kfold, scorers, refit=True):  # cv_how='grid_search',
  def minority_class_size(y):
    return min(Counter(y).values())
  assert kfold >= 1, 'kfold must be a positive integer!'

  n_frts = X.shape[1]
  minority_size = minority_class_size(y)
  min_samples_split_max = int(minority_size * (1-1/kfold))
  print("Minority-class size: ", minority_size)
  if md == globals.LGR:
    param_space = {'Cs': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                   'l1_ratio': [0, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99, 1],
                   'class_weight': [None, 'balanced']}
    clf = LogisticRegression(random_state=globals.SEED, penalty='elasticnet', solver='saga', max_iter=300)
  elif md == globals.SVC:
    clf = SVC(random_state=globals.SEED, probability=False)
    param_space = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                   'gamma': sorted(list({1 / n_frts, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001})),
                   'kernel': ['rbf', 'poly', 'sigmoid'],
                   'degree': [2, 3, 4],
                   'class_weight': [None, 'balanced']}
  elif md == globals.KNN:
    clf = KNeighborsClassifier()
    param_space = {
      'algorithm': ['ball_tree', 'kd_tree', 'brute'],
      'leaf_size': list(range(20, minority_size, 10)),
      'metric': ['minkowski'],
      'n_neighbors': list(range(5, minority_size + 1, minority_size // 10 + 1)),
      'p': [1, 2, 3],
      'weights': ['uniform', 'distance']
    }
  elif md == globals.DTCLF:
    clf = DecisionTreeClassifier(random_state=globals.SEED)
    param_space = {
      'class_weight': [None, 'balanced'],
      'max_depth': [None] + list(range(2, 21, 2)),
      'max_features': list(range(2, 1 + n_frts // 2, 10)) + [n_frts],
      'max_leaf_nodes': [None] + list(range(5, 101, 5)),
      'max_samples': np.arange(0.1, 1, 0.1),
      'min_samples_leaf': [1] + [i for i in range(2, 17, 2)],
      'min_samples_split': list(range(2, min_samples_split_max, 3)),
      'splitter': ['best', 'random'],
    }
  elif md == globals.RMFCLF:
    clf = RandomForestClassifier(random_state=globals.SEED)
    param_space = {
      'class_weight': [None, 'balanced'],
      'max_depth': [None] + list(range(2, 21, 2)),
      'max_features': sorted(set(list(range(2, n_frts // 2 + 1, n_frts // 20 + 1)) + [int(np.sqrt(n_frts))])),
      'max_leaf_nodes': [None] + list(range(5, 101, 5)),
      'max_samples': np.arange(0.1, 1, 0.1),
      'min_samples_leaf': [1] + [i for i in range(2, 17, 2)],
      'min_samples_split': list(range(2, min_samples_split_max, 3)),
      'n_estimators': [20, 30, 40, 50, 80, 100, 200, 300, 400]
    }
    if not param_space['min_samples_split']:
      raise ValueError("min_samples_split cannot < 2!")
  elif md == globals.GBCLF:
    # TODO: what's validation fraction?
    clf = GradientBoostingClassifier(random_state=globals.SEED, validation_fraction=0.15, n_iter_no_change=3)
    param_space = {
      'class_weight': [None, 'balanced'],
      'learning_rate': [0.001, 0.03, 0.01, 0.3, 0.1, 0.3],
      'loss': 'deviance',
      'max_depth': [None] + list(range(2, 21, 2)),
      'max_features': [None] + list(range(2, 1 + n_frts // 2, 2)),
      'max_leaf_nodes': [None] + list(range(5, )),
      'min_samples_leaf': [1, 2, 3, 4, 5],
      'min_samples_split': list(range(2, min_samples_split_max, 3)),
      'n_estimators': [20, 30, 40, 50, 80, 100, 200, 300, 400]
    }
    if not param_space['min_samples_split']:
      raise ValueError("min_samples_split cannot < 2!")
  elif md == globals.XGBCLF:
    clf = xgboost.XGBClassifier(random_state=globals.SEED)
    if n_frts > 500:
      colsample_bytree_range = np.arange(0.3, 0.9, 0.1)
    else:
      colsample_bytree_range = np.arange(0.8, 1.01, 0.05)
    # ?? define objective?? multi:softmax ???
    param_space = {
      'n_estimators': np.arange(50, 301, 50),
      'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.2, 0.3],
      'subsample': np.arange(0.8, 1.01, 0.05),
      'colsample_bytree': colsample_bytree_range,
      'max_depth': [3, 4, 5, 6, 7],
      'gamma': [0, 0.1, 0.3, 1, 3, 5],
      'min_child_weight': [0.1, 0.3, 0.6, 1, 2],
    }
  else:
    raise NotImplementedError("Model %s is not supported!" % md)

  # Use GridSearchCV for hyperparameter tuning
  grid_search = GridSearchCV(estimator=clf, param_grid=param_space, scoring=scorers, cv=kfold, n_jobs=-1,
                             refit=refit, return_train_score=True, verbose=0)  # TODO: n_jobs??
  grid_search.fit(X, y)
  return grid_search

## XGB other params:
# 'reg_alpha': None,  # L1 reg - faster under high dimensionality
# 'reg_lambda': None,  # L2 regularization - reduce overfitting
# 'scale_pos_weight': None,  # val > 0 to enable faster convergence under imbalanced class
# 'tree_method': None,
# 'objective': 'binary:logistic',
# 'use_label_encoder': True,
# 'base_score': None,
# 'booster': None,
# 'colsample_bylevel': None,
# 'colsample_bynode': None,
# 'gpu_id': None,
# 'importance_type': 'gain',
# 'interaction_constraints': None,
# 'max_delta_step': None,
# 'monotone_constraints': None,
# 'n_jobs': None,
# 'num_parallel_tree': None,