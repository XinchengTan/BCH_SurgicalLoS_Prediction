# Customized Model classes
import numpy as np
import optuna
from collections import Counter
from tqdm import tqdm
from typing import List

from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PoissonRegressor, Ridge, RidgeCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBClassifier, XGBRegressor

from globals import *
from c1_data_preprocessing import Dataset


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


# CV wrapper for KNN (grid search)
class KNeighborsClassifierCV(ClassifierCV):

  def __init__(self, param_space, kfold=5):
    super().__init__()
    self.clf = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_space, scoring='accuracy', cv=kfold,
                            n_jobs=-1, refit=True, return_train_score=True, verbose=0)


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

  def predict_proba(self, X):
    if len(self.classes_) == 1:
      return np.ones((X.shape[1], 1))
    return self.base_estimator.predict_proba(X)

  def predict(self, X):
    if len(self.classes_) == 1:
      return np.full(X.shape[0], self.classes_[0])
    return self.base_estimator.predict(X)


def get_model(model, cls_weight='balanced'):
  if model == LGR:
    clf = LogisticRegression(C=0.03, class_weight=cls_weight, max_iter=500, random_state=0)
    # clf = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], class_weight=cls_weight, max_iter=500, random_state=0,
    #                            cv=5)
  elif model == PR:
    clf = RegressionBasedClassifier(PR, alpha=0.01, max_iter=300)
  elif model == SVC:
    clf = SVC(gamma='auto', class_weight=cls_weight, probability=False)
  elif model == KNN:
    clf = KNeighborsClassifier(n_neighbors=25)
  elif model == KNNCV:
    clf = KNeighborsClassifierCV({'n_neighbors': np.arange(5, 41, 2)})
  elif model == DTCLF:
    clf = DecisionTreeClassifier(random_state=0, max_depth=4, class_weight=cls_weight)
  elif model == RMFCLF:
    clf = RandomForestClassifier(random_state=0, class_weight=cls_weight)
  elif model == GBCLF:
    clf = GradientBoostingClassifier(random_state=0, max_depth=3)
  elif model == XGBCLF:
    clf = XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=150, random_state=0, use_label_encoder=False,
                        eval_metric='mlogloss')
  elif model == ORDCLF_LOGIT:
    clf = OrdinalClassifier(distr='logit', solver='bfgs', disp=True)
  elif model == ORDCLF_PROBIT:
    clf = OrdinalClassifier(distr='probit', solver='bfgs', disp=True)
  elif model == BAL_BAGCLF:
    clf = BalancedBaggingClassifier(random_state=0)
  else:
    raise NotImplementedError("Model %s is not supported!" % model)

  return SafeOneClassWrapper(clf)


def get_model_by_cohort(model, cls_weight='balanced', cohort=None):
  # TODO: can add customization for each cohort later
  if model == LGR:
    clf = LogisticRegression(C=0.03, class_weight=cls_weight, max_iter=500, random_state=0)
    # clf = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], class_weight=cls_weight, max_iter=500, random_state=0,
    #                            cv=5)
  elif model == PR:
    clf = RegressionBasedClassifier(PR, alpha=0.01, max_iter=300)
  elif model == SVC:
    clf = SVC(gamma='auto', class_weight=cls_weight, probability=False)
  elif model == KNN:
    clf = KNeighborsClassifierCV({'n_neighbors': np.arange(4, 16)})
  elif model == DTCLF:
    clf = DecisionTreeClassifier(random_state=0, max_depth=4, class_weight=cls_weight)
  elif model == RMFCLF:
    clf = RandomForestClassifier(random_state=0, class_weight=cls_weight)
  elif model == GBCLF:
    clf = GradientBoostingClassifier(random_state=0, max_depth=3)
  elif model == XGBCLF:
    clf = XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, gamma=5, random_state=0,
                        use_label_encoder=False, eval_metric='mlogloss')
  elif model == ORDCLF_LOGIT:
    clf = OrdinalClassifier(distr='logit', solver='bfgs', disp=True)
  elif model == ORDCLF_PROBIT:
    clf = OrdinalClassifier(distr='probit', solver='bfgs', disp=True)
  elif model == BAL_BAGCLF:
    clf = BalancedBaggingClassifier(random_state=0)
  else:
    raise NotImplementedError("Model %s is not supported!" % model)

  return SafeOneClassWrapper(clf)


def train_model(md, params, X, y):
  if md == LGR:
    clf = LogisticRegression(random_state=SEED)
  elif md == SVC:
    clf = SVC(random_state=SEED, probability=True)
  elif md == KNN:
    clf = KNeighborsClassifier()
  elif md == DTCLF:
    clf = DecisionTreeClassifier(random_state=SEED)
  elif md == RMFCLF:
    clf = RandomForestClassifier(random_state=SEED)
  elif md == GBCLF:
    clf = GradientBoostingClassifier(random_state=SEED)
  elif md == XGBCLF:
    clf = XGBClassifier(random_state=SEED, eval_metric='mlogloss')
  else:
    raise NotImplementedError(f"Model {md} is not implemented yet!")

  clf.set_params(**params)
  clf = SafeOneClassWrapper(clf)
  clf.fit(X, y)
  return clf


def train_model_all_ktrials(decileFtr_config, models, k_datasets: List[Dataset], train_sda_only=False):
  # print decile agg funcs
  for k, v in decileFtr_config.items():
    print(k, v)

  models = [LGR, KNN, RMFCLF, XGBCLF] if models is None else models  # GBCLF,
  k_model_dict = []
  for kt, dataset_k in tqdm(enumerate(k_datasets), total=len(k_datasets)):
    # Fit models
    Xtrain, ytrain = dataset_k.get_sda_Xytrain() if train_sda_only else dataset_k.Xtrain, dataset_k.ytrain
    model_dict = {}
    for md in models:
      print('md=', md)
      clf = get_model(md)
      clf.fit(Xtrain, ytrain)
      model_dict[md] = clf
    k_model_dict.append(model_dict)
  return k_model_dict


# TODO: use Optuna
def train_model_cv(md, X, y, kfold, scorers, refit=True):  # cv_how='grid_search',
  def minority_class_size(y):
    return min(Counter(y).values())
  assert kfold >= 1, 'kfold must be a positive integer!'

  n_frts = X.shape[1]
  minority_size = minority_class_size(y)
  min_samples_split_max = int(minority_size * (1-1/kfold))
  print("Minority-class size: ", minority_size)
  if md == LGR:
    param_space = {'Cs': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                   'l1_ratio': [0, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99, 1],
                   'class_weight': [None, 'balanced']}
    clf = LogisticRegression(random_state=SEED, penalty='elasticnet', solver='saga', max_iter=300)
  elif md == SVC:
    clf = SVC(random_state=SEED, probability=False)
    param_space = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                   'gamma': sorted(list({1 / n_frts, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001})),
                   'kernel': ['rbf', 'poly', 'sigmoid'],
                   'degree': [2, 3, 4],
                   'class_weight': [None, 'balanced']}
  elif md == KNN:
    clf = KNeighborsClassifier()
    param_space = {
      'algorithm': ['ball_tree', 'kd_tree', 'brute'],
      'leaf_size': list(range(20, minority_size, 10)),
      'metric': ['minkowski'],
      'n_neighbors': list(range(5, minority_size + 1, minority_size // 10 + 1)),
      'p': [1, 2, 3],
      'weights': ['uniform', 'distance']
    }
  elif md == DTCLF:
    clf = DecisionTreeClassifier(random_state=SEED)
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
  elif md == RMFCLF:
    clf = RandomForestClassifier(random_state=SEED)
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
  elif md == GBCLF:
    # TODO: what's validation fraction?
    clf = GradientBoostingClassifier(random_state=SEED, validation_fraction=0.15, n_iter_no_change=3)
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
  elif md == XGBCLF:
    clf = XGBClassifier(random_state=SEED)
    if n_frts > 500:
      colsample_bytree_range = np.arange(0.3, 0.9, 0.1)
    else:
      colsample_bytree_range = np.arange(0.8, 1.01, 0.05)
    param_space = {
      'n_estimators': np.arange(50, 301, 50),
      'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.2, 0.3],
      'subsample': np.arange(0.8, 1.01, 0.05),
      'colsample_bytree': colsample_bytree_range,
      'max_depth': [3, 4, 5, 6, 7],
      'gamma': [0, 0.1, 0.3, 1, 3, 5],
      'min_child_weight': [0.1, 0.3, 0.6, 1, 2],
      'eval_metric': 'mlogloss',
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
# 'tree_method': None
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