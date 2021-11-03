"""
Predictive models for LoS
- regression models
- SVM (start with default rbf kernel, monitor train/val loss)
- Decision Trees, Random Forest, XGBoost, CART?
- KNN?

Preprocessing:
- normalization
- one-hot encode categorical variables
- mixed-type inputs VS all discrete variables

Input variables:
- demographics info
- diagnosis codes
- commorbidity ?= decile

Output type:
- LoS (continuous)
- range of LoS (discrete & ordinal), e.g. LoS in [0, 1), [1, 2), [2,3), [3, 4) ...
(-- resp, cardio outcome)

Evaluation:
- Confusion matrix
- ROC curve

Analysis:
- feature importance (Shapley value)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, make_scorer, plot_roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from statsmodels.miscmodels.ordinal_model import OrderedModel
from xgboost import XGBClassifier

from . import globals
from . import model_eval
from . import data_preprocessing as dpp
from . import plot_utils as pltutil


class AllModels(object):

  def __init__(self, regs_map=None, multi_clfs_map=None, bin_cutoff2clfs=None):
    self.mdname_model_map = dict()
    self.mdnames = []
    if regs_map is not None:
      for mdname, model in regs_map.items():
        self.mdname_model_map["REG_" + mdname] = model
        self.mdnames.append("REG_" + mdname)
    if multi_clfs_map is not None:
      for mdname, model in multi_clfs_map.items():
        self.mdname_model_map["MULTI_CLF\n" + mdname] = model
        self.mdnames.append("MULTI_CLF\n" + mdname)
    if bin_cutoff2clfs is not None:
      for mdname, cutoff2clf in bin_cutoff2clfs.items():
        for cutoff, clf in cutoff2clf.items():
          name = "< %d NNTs\n(%s)" % (cutoff, mdname)
          self.mdname_model_map[name] = clf
          self.mdnames.append(name)

  def predict_all(self, X):
    if len(self.mdname_model_map) == 0:
      return None
    md2preds = dict()
    for mdname, model in self.mdname_model_map.items():
      md2preds[mdname] = model.predict(X)
    return md2preds


def run_regression_model(Xtrain, ytrain, Xtest, ytest, model=globals.LR, eval=True):
  if model == globals.LR:
    reg = LinearRegression().fit(Xtrain, ytrain)
  elif model == globals.RIDGECV:
    reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(Xtrain, ytrain)
  elif model == globals.DT:
    reg = DecisionTreeRegressor(random_state=0, max_depth=4).fit(Xtrain, ytrain)
  elif model == globals.RMF:
    reg = RandomForestRegressor(max_depth=5, random_state=0).fit(Xtrain, ytrain)
  elif model == globals.GB:
    reg = GradientBoostingRegressor(random_state=0, max_depth=2).fit(Xtrain, ytrain)
  else:
    raise ValueError("Model %s not supported!" % model)

  if eval:
    pred_train, pred_val = reg.predict(Xtrain), reg.predict(Xtest)
    train_mse = mean_squared_error(ytrain, pred_train)
    val_mse = mean_squared_error(ytest, pred_val)
    print("Raw regressor %s:" % globals.reg2name[model])
    print("R-squared (training set): ", reg.score(Xtrain, ytrain))
    print("R-squared (validation set): ", reg.score(Xtest, ytest))
    print("MSE (training set): ", train_mse, "RMSE: ", np.sqrt(train_mse))
    print("MSE (validation set): ", val_mse, "RMSE: ", np.sqrt(val_mse))
    print("\n")
  return reg


def predict_los(Xtrain, ytrain, Xtest, ytest, model=None, isTest=False):
  model2trained = {}
  if model is None:  # run all
    for md, md_name in globals.reg2name.items():
      reg = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=md)
      model2trained[model] = reg
      pred_train, pred_test = reg.predict(Xtrain), reg.predict(Xtest)

      # Error histogram
      figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
      pltutil.plot_error_histogram(ytrain, pred_train, md_name, Xtype='train', yType="Number of nights",
                                     ax=axs[0])
      pltutil.plot_error_histogram(ytest, pred_test, md_name, Xtype='validation', yType="Number of nights",
                                     ax=axs[1])
  else:  # run a specific model
    reg = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=model)
    model2trained[model] = reg

  return model2trained


def run_classifier(Xtrain, ytrain, model, cls_weight=None):
  # TODO: Need to use cross validation for model selection here, consider eval metric other than built-in criteria
  if model == globals.LGR:
    clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, class_weight=cls_weight, max_iter=300)).fit(Xtrain, ytrain)
  elif model == globals.SVC:  # TODO: default is rbf kernel, need to experiment with others
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', class_weight=cls_weight, probability=True)).fit(Xtrain, ytrain)
  elif model == globals.DTCLF:
    clf = DecisionTreeClassifier(random_state=0, max_depth=4, class_weight=cls_weight).fit(Xtrain, ytrain)
  elif model == globals.RMFCLF:
    clf = RandomForestClassifier(max_depth=6, random_state=0, class_weight=cls_weight).fit(Xtrain, ytrain)
  elif model == globals.GBCLF:
    clf = GradientBoostingClassifier(random_state=0, max_depth=2).fit(Xtrain, ytrain)  # TODO: imbalanced class issue here!
  elif model == globals.XGBCLF:
    clf = XGBClassifier(random_state=0).fit(Xtrain, ytrain)
  else:
    raise NotImplementedError("Model %s is not supported!" % model)

  return clf


def run_classifier_cv(X, y, md, scorer, class_weight=None, kfold=5):
  n_frts = X.shape[1]
  if md == globals.SVC:
    pass
  elif md == globals.DTCLF:
    pass
  elif md == globals.RMFCLF:
    #     'oob_score': False
    #     'min_impurity_decrease': 0.0,
    #     'min_impurity_split': None,
    #     'criterion': ['gini', 'entropy'],
    #     'bootstrap': True
    clf = RandomForestClassifier(random_state=globals.SEED, class_weight=class_weight)
    param_space = {
      'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      'max_features': np.arange(2, 1 + n_frts // 2),
      'max_leaf_nodes': [None] + list(range(5, 101, 5)),
      'max_samples': np.arange(0.1, 1, 0.1),
      'min_samples_leaf': [1] + [i for i in range(2, 17, 2)],
      'min_samples_split': np.arange(2, 65, 4),
      'n_estimators': [20, 30, 40, 50, 80, 100, 200, 300, 400]
    }
  elif md == globals.GBCLF:
    clf = GradientBoostingClassifier(random_state=globals.SEED, class_weight=class_weight)
    param_space = {
      'learning_rate': [0.001, 0.03, 0.01, 0.3, 0.1, 0.3],
      'loss': 'deviance',
      'max_depth': np.arange(3, 10),
      'max_features': np.arange(2, n_frts+1, 2),
      'max_leaf_nodes': None,
      #'min_impurity_decrease': 0.0,
      #'min_impurity_split': None,
      # 'min_weight_fraction_leaf': 0.0,
      'min_samples_leaf': 1,
      'min_samples_split': 2,
      'n_estimators': [20, 30, 40, 50, 80, 100, 200, 300, 400]
    }
  else:
    raise NotImplementedError("Model %s is not supported!" % md)

  # Define scorer
  refit = True
  if scorer == globals.SCR_1NNT_TOL:
    scorer = make_scorer(scorer_1nnt_tol, greater_is_better=True)
  elif scorer == globals.SCR_1NNT_TOL_ACC:
    scorer = {globals.SCR_ACC: globals.SCR_ACC,
              globals.SCR_1NNT_TOL: make_scorer(scorer_1nnt_tol, greater_is_better=True)}
    refit = globals.SCR_1NNT_TOL
  elif scorer == globals.SCR_MULTI_ALL:
    scorer = {globals.SCR_ACC: globals.SCR_ACC, globals.SCR_AUC: globals.SCR_AUC,
              globals.SCR_1NNT_TOL: make_scorer(scorer_1nnt_tol, greater_is_better=True)}
    refit = globals.SCR_AUC

  # For each parameter, iterate through its param grid
  param2gs = {}
  for param, param_grid in param_space.items():
    print("\nSearching %s among " % param, param_grid)
    gs = GridSearchCV(estimator=clf, param_grid={param: param_grid}, scoring=scorer, n_jobs=-1, cv=kfold,
                      refit=refit, return_train_score=True, verbose=2)
    gs.fit(X, y)
    param2gs[param] = gs
  return param2gs, param_space


def scorer_1nnt_tol(ytrue, ypred):
  # accuracy within +-1 nnt error tolerance
  acc_1nnt_tol = len(np.where(np.abs(ytrue - ypred) <= 1)[0]) / len(ytrue)
  return acc_1nnt_tol


def predict_nnt_regression_rounding(Xtrain, ytrain, Xval, yval, model=None, Xtest=None, ytest=None):
  """
  Predict number of nights via regression & rounding to nearest int
  """
  model2trained = {}
  if model is None:  # run all
    for md, md_name in globals.reg2name.items():
      reg = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=md, eval=True)
      model2trained[md] = reg
      model_eval.eval_nnt_regressor(reg, Xtrain, ytrain, Xval, yval, md_name)
  else:
    reg = run_regression_model(Xtrain, ytrain, Xval, yval, model=model)
    model2trained[model] = reg
    model_eval.eval_nnt_regressor(reg, Xtrain, ytrain, Xval, yval, md_name=globals.reg2name[model])

  return model2trained


def predict_nnt_multi_clf(Xtrain, ytrain, Xval, yval, model=None, cls_weight=None, eval=True, Xtest=None, ytest=None):
  """ Predict number of nights via a multi-class classfier"""
  model2trained = {}
  model2f1s = {}
  if model is None:
    if not eval:
      for md, md_name in globals.clf2name.items():
        print("Fitting %s" % md_name)
        clf = run_classifier(Xtrain, ytrain, model=md, cls_weight=cls_weight)
        model2trained[md] = clf
    else:
      for md, md_name in globals.clf2name.items():
        clf = run_classifier(Xtrain, ytrain, model=md, cls_weight=cls_weight)
        model2trained[md] = clf
        _, _, f1_train, f1_val = model_eval.eval_multi_clf(clf, Xtrain, ytrain, Xval, yval, md_name)
        model2f1s[md] = [f1_train, f1_val]
  else:
    clf = run_classifier(Xtrain, ytrain, model=model, cls_weight=cls_weight)
    model2trained[model] = clf
    _, _, f1_train, f1_val = model_eval.eval_multi_clf(clf, Xtrain, ytrain, Xval, yval, md_name=globals.clf2name[model])
    model2f1s[model] = [f1_train, f1_val]

  return model2trained, model2f1s


def predict_nnt_binary_clf(Xtrain, ytrain, Xval, yval, clf_cutoffs, metric=None, model=None, cls_weight=None, eval=True,
                           Xtest=None, ytest=None):
  """
  Predict number of nights via a series of binary classifiers, given a list of integer cutoffs.
  e.g. [1, 2] means two classifiers: one for if NNT < 1 or >= 1, the other for if NNT < 2 or >= 2
  """
  model2binclfs = {}
  if model is None:
    # fit multiple binary classifiers independently
    # assume classifier_cutoffs is an increasing list of integers, each represent a cutoff value
    if not eval:
      for md, md_name in globals.clf2name.items():
        print("Fitting binary classifiers %s" % md_name)
        cutoff2clf = {}
        for cutoff in clf_cutoffs:
          # Build binary outcome
          ytrain_b = dpp.gen_y_nnt_binary(ytrain, cutoff)
          # Train classifier
          clf = run_classifier(Xtrain, ytrain_b, model=md, cls_weight=cls_weight)
          cutoff2clf[cutoff] = clf
        model2binclfs[md] = cutoff2clf  # save models
    else:
      for cutoff in clf_cutoffs:
        figs, axs = plt.subplots(ncols=2, nrows=1, figsize=(20, 7))
        cutoff2clf = {}
        for md, md_name in globals.clf2name.items():
          # Build binary outcome
          ytrain_b = dpp.gen_y_nnt_binary(ytrain, cutoff)
          yval_b = dpp.gen_y_nnt_binary(yval, cutoff)

          # Train classifier
          clf = run_classifier(Xtrain, ytrain_b, model=md, cls_weight=cls_weight)
          cutoff2clf[cutoff] = clf

          # Evaluate model
          if eval:
            pred_train, pred_val, ix = model_eval.eval_binary_clf(clf, cutoff, Xtrain, ytrain_b, Xval, yval_b, md_name,
                                                                  metric=metric, plot_roc=False)

            # Generate feature importance ranking
            model_eval.gen_feature_importance_bin_clf(clf, md, Xval, yval_b, cutoff=cutoff)

          # Plot ROC curve for the current model at 'cutoff'
          plot_roc_curve(clf, Xtrain, ytrain_b, name=md_name, ax=axs[0])
          plot_roc_curve(clf, Xval, yval_b, name=md_name, ax=axs[1])
          if metric == globals.GMEAN:
            pltutil.plot_roc_best_threshold(Xtrain, ytrain_b, clf, axs[0])
            pltutil.plot_roc_best_threshold(Xval, yval_b, clf, axs[1])
          elif metric == globals.FPRPCT15:
            continue
          # TODO: add prec-recall plot for F1

        pltutil.plot_roc_basics(axs[0], cutoff, 'training')
        pltutil.plot_roc_basics(axs[1], cutoff, 'validation')
        plt.show()
        model2binclfs[model] = cutoff2clf  # save models
  else:
    cutoff2clf = {}
    figs, axs = plt.subplots(nrows=4, ncols=4, figsize=(21, 21))
    for cutoff in clf_cutoffs:
      # Build binary outcome
      ytrain_b = dpp.gen_y_nnt_binary(ytrain, cutoff)
      yval_b = dpp.gen_y_nnt_binary(yval, cutoff)

      # Train classifier
      clf = run_classifier(Xtrain, ytrain_b, model=model, cls_weight=cls_weight)
      cutoff2clf[cutoff] = clf

      # Evaluate model
      model_eval.eval_binary_clf(clf, cutoff, Xtrain, ytrain_b, Xval, yval_b, globals.clf2name[model], metric=metric,
                                 plot_roc=False, axs=[axs[(cutoff-1)//4][(cutoff-1)%4], axs[2+(cutoff-1)//4][(cutoff-1)%4]])

      # Generate feature importance ranking
      #model_eval.gen_feature_importance_bin_clf(clf, model, Xval, yval_b, cutoff=cutoff)
    model2binclfs[model] = cutoff2clf
    figs.tight_layout()
    figs.savefig("%s (val-%s) binclf.png" % (model, str(cls_weight)))

  return model2binclfs

