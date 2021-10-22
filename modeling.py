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
from sklearn.metrics import mean_squared_error, roc_curve, plot_roc_curve

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

from . import globals
from . import model_eval


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
      model_eval.gen_error_histogram(ytrain, pred_train, md_name, Xtype='train', yType="Number of nights",
                                     ax=axs[0])
      model_eval.gen_error_histogram(ytest, pred_test, md_name, Xtype='validation', yType="Number of nights",
                                     ax=axs[1])
  else:  # run a specific model
    reg = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=model)
    model2trained[model] = reg

  return model2trained


def run_classifier(Xtrain, ytrain, model):
  # TODO: Need to use cross validation for model selection here, consider eval metric other than built-in criteria
  if model == globals.LGR:
    clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, class_weight='balanced')).fit(Xtrain, ytrain)
  elif model == globals.SVC:  # TODO: default is rbf kernel, need to experiment with others
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', class_weight='balanced', probability=True)).fit(Xtrain, ytrain)
  elif model == globals.DTCLF:
    clf = DecisionTreeClassifier(random_state=0, max_depth=4, class_weight='balanced').fit(Xtrain, ytrain)
  elif model == globals.RMFCLF:
    clf = RandomForestClassifier(max_depth=6, random_state=0, class_weight='balanced').fit(Xtrain, ytrain)
  elif model == globals.GBCLF:
    clf = GradientBoostingClassifier(random_state=0, max_depth=2).fit(Xtrain, ytrain)  # TODO: imbalanced class issue here!
  else:
    raise NotImplementedError("Model %s is not supported!" % model)

  return clf


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


def predict_nnt_multi_clf(Xtrain, ytrain, Xval, yval, model=None, Xtest=None, ytest=None):
  """ Predict number of nights via a multi-class classfier"""
  model2trained = {}
  if model is None:
    for md, md_name in globals.clf2name.items():
      clf = run_classifier(Xtrain, ytrain, model=md)
      model2trained[md] = clf
      model_eval.eval_multi_clf(clf, Xtrain, ytrain, Xval, yval, md_name)
  else:
    clf = run_classifier(Xtrain, ytrain, model=model)
    model2trained[model] = clf
    model_eval.eval_multi_clf(clf, Xtrain, ytrain, Xval, yval, md_name=globals.clf2name[model])

  return model2trained


def predict_nnt_binary_clf(Xtrain, ytrain, Xval, yval, clf_cutoffs, metric=None, model=None, eval=True, Xtest=None, ytest=None):
  """
  Predict number of nights via a series of binary classifiers, given a list of integer cutoffs.
  e.g. [1, 2] means two classifiers: one for if NNT < 1 or >= 1, the other for if NNT < 2 or >= 2
  """
  model2binclfs = {}
  if model is None:
    # fit multiple binary classifiers independently
    # assume classifier_cutoffs is an increasing list of integers, each represent a cutoff value
    for cutoff in clf_cutoffs:
      figs, axs = plt.subplots(ncols=2, nrows=1, figsize=(20, 7))
      cutoff2clf = {}
      for md, md_name in globals.clf2name.items():
        # Build binary outcome
        ytrain_b = np.copy(ytrain)
        ytrain_b[ytrain_b < cutoff] = 0
        ytrain_b[ytrain_b >= cutoff] = 1
        yval_b = np.copy(yval)
        yval_b[yval_b < cutoff] = 0
        yval_b[yval_b >= cutoff] = 1

        # Train classifier
        clf = run_classifier(Xtrain, ytrain_b, model=md)
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
          model_eval.plot_roc_best_threshold(Xtrain, ytrain_b, clf, axs[0])
          model_eval.plot_roc_best_threshold(Xval, yval_b, clf, axs[1])
        elif metric == globals.FPRPCT15:

          continue
        # TODO: add prec-recall plot for F1

      model_eval.plot_roc_basics(axs[0], cutoff, 'training')
      model_eval.plot_roc_basics(axs[1], cutoff, 'validation')
      plt.show()
      model2binclfs[model] = cutoff2clf  # save models
  else:
    cutoff2clf = {}
    for cutoff in clf_cutoffs:
      # Build binary outcome
      ytrain_b = np.copy(ytrain)
      ytrain_b[ytrain_b < cutoff] = 0
      ytrain_b[ytrain_b >= cutoff] = 1
      yval_b = np.copy(yval)
      yval_b[yval_b < cutoff] = 0
      yval_b[yval_b >= cutoff] = 1

      # Train classifier
      clf = run_classifier(Xtrain, ytrain_b, model=model)
      cutoff2clf[cutoff] = clf

      # Evaluate model
      model_eval.eval_binary_clf(clf, cutoff, Xtrain, ytrain_b, Xval, yval_b, globals.clf2name[model], metric=metric, plot_roc=False)

      # Generate feature importance ranking
      model_eval.gen_feature_importance_bin_clf(clf, model, Xval, yval_b, cutoff=cutoff)
    model2binclfs[model] = cutoff2clf

  return model2binclfs


# def predict_nnt(Xtrain, ytrain, Xval, yval, Xtest, ytest, use_reg=True, clf_cutoffs=None, model=None):
#   """
#   Predictive models for number of nights stay
#
#   :param Xtrain:
#   :param ytrain:
#   :param Xtest:
#   :param ytest:
#   :param clf_cutoffs:
#   :param model:
#   :param isTest:
#   :return:
#   """
#   model2trained = {}
#
#   if model is None:  # run all
#     if clf_cutoffs is None:  # fit regression models and round to the nearest int
#       if use_reg:
#         for md, md_name in globals.reg2name.items():
#           reg = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=md, eval=True)
#           model2trained[md] = reg
#           # predict and round to the nearest int
#           pred_train, pred_test = np.rint(reg.predict(Xtrain)), np.rint(reg.predict(Xtest))
#           # bucket them into finite number of classes
#           pred_train[pred_train > globals.MAX_NNT] = globals.MAX_NNT + 1
#           pred_test[pred_test > globals.MAX_NNT] = globals.MAX_NNT + 1
#
#           print("Accuracy (training): ", accuracy_score(ytrain, pred_train, normalize=True))
#           print("Accuracy (validation): ", accuracy_score(ytest, pred_test, normalize=True))
#
#           # Confusion matrix
#           labels = [str(i) for i in range(globals.MAX_NNT + 2)]
#           labels[-1] = '%s+' % globals.MAX_NNT
#           model_eval.gen_confusion_matrix(ytrain, pred_train, md_name, isTrain=True)
#           model_eval.gen_confusion_matrix(ytest, pred_test, md_name, isTrain=False)
#
#           # Error histogram
#           figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
#           model_eval.gen_error_histogram(ytrain, pred_train, md_name, Xtype='train', yType="Number of nights",
#                                          ax=axs[0])
#           model_eval.gen_error_histogram(ytest, pred_test, md_name, Xtype='validation', yType="Number of nights",
#                                          ax=axs[1])
#       else:
#         for md, md_name in globals.clf2name.items():
#           clf = run_classifier(Xtrain, ytrain, model=md)
#           model2trained[md] = clf
#           pred_train, pred_test = model_eval.eval_classifier(clf, Xtrain, ytrain, Xtest, ytest, model=md)
#
#           # Confusion matrix
#           labels = [str(i) for i in range(globals.MAX_NNT + 2)]
#           labels[-1] = '%s+' % globals.MAX_NNT
#           model_eval.gen_confusion_matrix(ytrain, pred_train, md_name, isTrain=True)
#           model_eval.gen_confusion_matrix(ytest, pred_test, md_name, isTrain=False)
#
#           # Error histogram
#           figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
#           model_eval.gen_error_histogram(ytrain, pred_train, md_name, Xtype='train', yType="Number of nights",
#                                          ax=axs[0])
#           model_eval.gen_error_histogram(ytest, pred_test, md_name, Xtype='validation', yType="Number of nights",
#                                          ax=axs[1])
#
#
#     else:  # fit multiple binary classifiers independently
#       # assume classifier_cutoffs is an increasing list of integers, each represent a cutoff value
#       for cutoff in clf_cutoffs:
#         fig, ax = plt.subplots(figsize=(11, 9))
#         for md, md_name in globals.clf2name.items():
#           # Build binary outcome
#           ytrain_b = np.copy(ytrain)
#           ytrain_b[ytrain_b < cutoff] = 0
#           ytrain_b[ytrain_b >= cutoff] = 1
#           ytest_b = np.copy(ytest)
#           ytest_b[ytest_b < cutoff] = 0
#           ytest_b[ytest_b >= cutoff] = 1
#
#           # Train classifier
#           clf = run_classifier(Xtrain, ytrain_b, model=md)
#
#           # Evaluate model
#           #eval_bin_clf(clf, cutoff, Xtrain, ytrain_b, Xtest, ytest_b, model=md, plot_roc=False)
#
#           # Generate feature importance ranking
#           #gen_feature_importance_bin_clf(clf, md, Xtest, ytest_b, cutoff=cutoff)
#
#           # Plot ROC curve
#           plot_roc_curve(clf, Xtest, ytest_b, name=md_name, ax=ax)
#         ax.set_title("Cutoff=%d: ROC Curve Comparison" % cutoff, y=1.01, fontsize=18)
#         ax.set_xlabel("False Positive Rate", fontsize=15)
#         ax.set_ylabel("True Positive Rate", fontsize=15)
#         ax.legend(prop=dict(size=14))
#         plt.show()
#
#
#   else:  # run a specific model
#     if clf_cutoffs is None:
#       clf = run_classifier(Xtrain, ytrain, model=model)
#       model2trained[model] = clf
#       model_eval.eval_classifier(clf, Xtrain, ytrain, Xtest, ytest, model=model)
#     else:
#       # independent classifiers
#       # e.g. [1, 2] --> two classifiers: one for if NNT < 1 or >= 1, the other for if NNT < 2 or >= 2
#       for cutoff in clf_cutoffs:
#         # Build binary outcome
#         ytrain_b = np.copy(ytrain)
#         ytrain_b[ytrain_b < cutoff] = 0
#         ytrain_b[ytrain_b >= cutoff] = 1
#         ytest_b = np.copy(ytest)
#         ytest_b[ytest_b < cutoff] = 0
#         ytest_b[ytest_b >= cutoff] = 1
#
#         # Train classifier
#         clf = run_classifier(Xtrain, ytrain_b, model=model)
#
#         # Evaluate model
#         #eval_bin_clf(clf, cutoff, Xtrain, ytrain_b, Xtest, ytest_b, model=model, plot_roc=False)
#
#         # Generate feature importance ranking
#         model_eval.gen_feature_importance_bin_clf(clf, model, Xtest, ytest_b, cutoff=cutoff)
#
#
#   return model2trained