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
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, make_scorer, plot_roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from statsmodels.miscmodels.ordinal_model import OrderedModel
from xgboost import XGBClassifier

from . import globals, model_eval, surgeon
from . import data_preprocessing as dpp
from .data_preprocessing import Dataset
from .model_eval import ModelPerf
from . import plot_utils as pltutil


class AllModels(object):

  def __init__(self, regs_map=None, multi_clfs_map=None, bin_cutoff2clfs=None):
    self.mdname_model_map = dict()
    self.mdnames = []
    if multi_clfs_map is not None:
      for mdname, model in multi_clfs_map.items():
        self.mdname_model_map["MULTI_CLF_%s" % mdname] = model
        self.mdnames.append("MULTI_CLF_%s" % mdname)
    if bin_cutoff2clfs is not None:
      for mdname, cutoff2clf in bin_cutoff2clfs.items():
        for cutoff, clf in cutoff2clf.items():
          name = "< %d NNTs  %s" % (cutoff, mdname)
          self.mdname_model_map[name] = clf
          self.mdnames.append(name)

  def predict_all(self, X):
    if len(self.mdname_model_map) == 0:
      return None
    md2preds = dict()
    for mdname, model in self.mdname_model_map.items():
      md2preds[mdname] = model.predict(X)
    return md2preds


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


def run_classifier(Xtrain, ytrain, model, cls_weight=None, calibrate_method=None, calibrate_on_val=True, standardize=False):
  """
  :param Xtrain: a numpy array (n_cases, n_features)
  :param ytrain: a numpy array (n_cases, )
  :param model: abbreviation of the model name
  :param cls_weight: If None, do not account for class imbalance; If 'balanced', correct for class imbalance in model parameters
  :param calibrate_method: If None, do not calibrate; otherwise, use 'sigmoid' or 'isotonic' to calibrate classifier
  :param standardize: If True, standardize "Xtrain" along each feature axis
  :return:
  """
  Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=globals.SEED)

  if standardize:
    Xtrain = StandardScaler().fit_transform(Xtrain)

  if model == globals.LGR:
    clf = LogisticRegression(random_state=0, class_weight=cls_weight, max_iter=300).fit(Xtrain, ytrain)
  elif model == globals.SVC:
    clf =SVC(gamma='auto', class_weight=cls_weight, probability=True).fit(Xtrain, ytrain)
  elif model == globals.DTCLF:
    clf = DecisionTreeClassifier(random_state=0, max_depth=4, class_weight=cls_weight).fit(Xtrain, ytrain)
  elif model == globals.RMFCLF:
    clf = RandomForestClassifier(max_features=130, random_state=0, class_weight=cls_weight).fit(Xtrain, ytrain)
  elif model == globals.GBCLF:
    clf = GradientBoostingClassifier(random_state=0, max_depth=2).fit(Xtrain, ytrain)  # TODO: imbalanced class issue here!
  elif model == globals.XGBCLF:
    clf = XGBClassifier(random_state=0).fit(Xtrain, ytrain)
  elif model == globals.ORDCLF_LOGIT:
    clf = OrdinalClassifier(distr='logit', solver='bfgs', disp=True).fit(Xtrain, ytrain)
  elif model == globals.ORDCLF_PROBIT:
    clf = OrdinalClassifier(distr='probit', solver='bfgs', disp=True).fit(Xtrain, ytrain)
  else:
    raise NotImplementedError("Model %s is not supported!" % model)

  if calibrate_method is not None:
    if calibrate_on_val:
      clf = CalibratedClassifierCV(clf, method=calibrate_method, cv='prefit').fit(Xval, yval)
    else:
      clf = CalibratedClassifierCV(clf, method=calibrate_method, cv='prefit').fit(Xtrain, ytrain)

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
      'max_features': np.arange(2, 1 + n_frts // 2, 10),
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
    scorer = make_scorer(model_eval.scorer_1nnt_tol, greater_is_better=True)
  elif scorer == globals.SCR_1NNT_TOL_ACC:
    scorer = {globals.SCR_ACC: globals.SCR_ACC,
              globals.SCR_1NNT_TOL: make_scorer(model_eval.scorer_1nnt_tol, greater_is_better=True)}
    refit = globals.SCR_1NNT_TOL
  elif scorer == globals.SCR_MULTI_ALL:
    scorer = {globals.SCR_ACC: globals.SCR_ACC, globals.SCR_AUC: globals.SCR_AUC,
              globals.SCR_1NNT_TOL: make_scorer(model_eval.scorer_1nnt_tol, greater_is_better=True)}
    refit = globals.SCR_AUC

  # For each parameter, iterate through its param grid
  param2gs = {}
  for param, param_grid in param_space.items():
    print("\nSearching %s among " % param, param_grid)
    gs = GridSearchCV(estimator=clf, param_grid={param: param_grid}, scoring=scorer, n_jobs=-1, cv=kfold,
                      refit=refit, return_train_score=True, verbose=0)
    gs.fit(X, y)
    param2gs[param] = gs
  return param2gs, param_space


def predict_nnt_regression_rounding(dataset: Dataset, model=None):
  """
  Predict number of nights via regression & rounding to nearest int
  """
  Xtrain, ytrain, Xtest, ytest = dataset.Xtrain, dataset.ytrain, dataset.Xtest, dataset.ytest

  model2trained = {}
  if model is None:  # run all
    for md, md_name in globals.reg2name.items():
      reg = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=md, eval=True)
      model2trained[md] = reg
      model_eval.eval_nnt_regressor(reg, dataset, md_name)
  else:
    reg = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=model)
    model2trained[model] = reg
    model_eval.eval_nnt_regressor(reg, dataset, md_name=globals.reg2name[model])

  return model2trained


def predict_nnt_multi_clf(dataset: Dataset, model=None, cls_weight=None, evaluate=True):
  """ Predict number of nights via a multi-class classfier"""
  Xtrain, ytrain, Xtest, ytest = dataset.Xtrain, dataset.ytrain, dataset.Xtest, dataset.ytest
  model2trained = {}
  if model is None:
    for md, md_name in globals.clf2name.items():
      print("Fitting %s" % md_name)
      clf = run_classifier(Xtrain, ytrain, model=md, cls_weight=cls_weight)
      model2trained[md] = clf
  else:
    clf = run_classifier(Xtrain, ytrain, model=model, cls_weight=cls_weight)
    model2trained[model] = clf

  if evaluate:
    model2ModelPerf, eval_dfs = performance_eval_multiclfs(dataset, model2trained, XType=None)
    for eval_df in eval_dfs:
      display(eval_df)

  return model2trained


def performance_eval_multiclfs(dataset: Dataset, model2trained_clf, XType):
  """
  A high-level function that iterates through all trained multi-class classifiers and evaluate
  the performance of each on test or test and training set.
  Performance evaluation includes:
  - hit rate, i.e. accuracy
  - accuracy with 1 NNT error tolerance
  - mean squared error
  - confusion matrix (normalized by true class counts)
  - ......
  """
  Xtrain, ytrain, Xtest, ytest = dataset.Xtrain, dataset.ytrain, dataset.Xtest, dataset.ytest
  df = dataset.df
  md2ModelPerf = defaultdict(lambda: defaultdict(lambda: None))
  md2pproc_df = defaultdict(lambda: defaultdict(lambda: None))
  xtypes = [XType] if XType is not None else [globals.XTRAIN, globals.XTEST]

  for md, clf in model2trained_clf.items():
    if XType is None:  # evaluate both training and test set
      md2ModelPerf[md][globals.XTRAIN] = model_eval.eval_multiclf_on_Xy(clf, df.iloc[dataset.train_idx], Xtrain, ytrain,
                                                                        globals.clf2name[md], globals.XTRAIN)
      md2ModelPerf[md][globals.XTEST] = model_eval.eval_multiclf_on_Xy(clf, df.iloc[dataset.test_idx], Xtest, ytest,
                                                                       globals.clf2name[md], globals.XTEST)

    elif XType == globals.XTRAIN:
      md2ModelPerf[md][XType] = model_eval.eval_multiclf_on_Xy(clf, df.iloc[dataset.train_idx], Xtrain, ytrain,
                                                               globals.clf2name[md], XType)

    elif XType == globals.XTEST:
      md2ModelPerf[md][XType] = model_eval.eval_multiclf_on_Xy(clf, df.iloc[dataset.test_idx], Xtest, ytest,
                                                               globals.clf2name[md], XType)

    elif XType == globals.XAGREE:
      data_df, Xdata, ydata = surgeon.gen_surgeon_model_agree_df_and_Xydata(dataset, model2trained_clf, md, use_test=True)
      md2ModelPerf[md][XType] = model_eval.eval_multiclf_on_Xy(clf, data_df, Xdata, ydata, globals.clf2name[md], XType)

    elif XType == globals.XDISAGREE:
      data_df, Xdata, ydata = surgeon.gen_surgeon_model_disagree_df_and_Xydata(dataset, model2trained_clf, md,
                                                                            use_test=True)
      md2ModelPerf[md][XType] = model_eval.eval_multiclf_on_Xy(clf, data_df, Xdata, ydata, globals.clf2name[md], XType)

    else:
      raise NotImplementedError("XType '%s' is not implemented yet!" % XType)

  # Create a dataframe table to present model performance along all the evaluation metrics
  eval_dfs = []
  for xtype in xtypes:
    # create a data table from a dict
    md2MP_xtype_dict = {globals.clf2name[md]: md2ModelPerf[md][xtype].get_perf_as_dict() for md in md2ModelPerf}
    df = pd.DataFrame.from_dict(md2MP_xtype_dict, orient='index').sort_values(by=["Accuracy"], ascending=False)
    df.index.name = 'Model'
    df = df.style\
      .set_table_attributes("style='display:inline'")\
      .set_caption("Model Performance (%s cases)" % xtype)\
      .format(ModelPerf.get_metrics_formatter())\
      .set_properties(**{'text-align': 'center'})
    eval_dfs.append(df)

  return md2ModelPerf, eval_dfs


def predict_nnt_binary_clf(dataset: Dataset, cutoffs, metric=None, model=None, cls_weight=None, eval=True,
                           calibrate_method=None, calibrate_on_val=True):
  """
  Predict number of nights via a series of binary classifiers, given a list of integer cutoffs.
  e.g. [1, 2] means two classifiers: one for if NNT < 1 or >= 1, the other for if NNT < 2 or >= 2
  """
  Xtrain, Xtest = dataset.Xtrain, dataset.Xtest
  ytrain, ytest = np.copy(dataset.ytrain), np.copy(dataset.ytest)

  model2binclfs = {}

  if model is None:
    # fit multiple binary classifiers independently
    # assume classifier_cutoffs is an increasing list of integers, each represent a cutoff value
    for md, md_name in globals.binclf2name.items():
      print("Fitting binary classifiers %s" % md_name)
      cutoff2clf = {}
      for cutoff in cutoffs:
        # Train classifier
        clf = run_classifier(Xtrain, dpp.gen_y_nnt_binary(ytrain, cutoff), model=md, cls_weight=cls_weight,
                             calibrate_method=calibrate_method, calibrate_on_val=calibrate_on_val)
        cutoff2clf[cutoff] = clf
      model2binclfs[md] = cutoff2clf  # save models

    if eval:
      run_all_eval_binclf(dataset, cutoffs, model2binclfs, metric)

  else:
    # When running each model individually, always evaluate
    cutoff2clf = {}
    figs, axs = plt.subplots(nrows=4, ncols=4, figsize=(21, 21))
    for cutoff in cutoffs:
      dataset.ytrain = dpp.gen_y_nnt_binary(ytrain, cutoff)
      dataset.ytest = dpp.gen_y_nnt_binary(ytest, cutoff)
      # Train classifier
      clf = run_classifier(Xtrain, dataset.ytrain, model=model, cls_weight=cls_weight, calibrate_method=calibrate_method)
      cutoff2clf[cutoff] = clf

      # Evaluate model
      model_eval.eval_binary_clf(clf, cutoff, dataset, globals.binclf2name[model], metric=metric,
                                 plot_roc=False, axs=[axs[(cutoff-1)//4][(cutoff-1)%4], axs[2+(cutoff-1)//4][(cutoff-1)%4]])

      # Generate feature importance
      #model_eval.gen_feature_importance_bin_clf(clf, model, Xval, yval_b, cutoff=cutoff)
    model2binclfs[model] = cutoff2clf
    figs.tight_layout()
    figs.savefig("%s (val-%s) binclf.png" % (model, str(cls_weight)))
    dataset.ytrain = ytrain
    dataset.ytest = ytest
  return model2binclfs


def run_all_eval_binclf(dataset, clf_cutoffs, model2binclfs, metric=None):
  Xtrain, Xtest = dataset.Xtrain, dataset.Xtest
  ytrain, ytest = np.copy(dataset.ytrain), np.copy(dataset.ytest)

  for cutoff in clf_cutoffs:
    figs, axs = plt.subplots(ncols=2, nrows=1, figsize=(20, 7))  # ROC curves
    figs2, axs2 = plt.subplots(ncols=2, nrows=1, figsize=(20, 7))  # Calibration curves
    for md, md_name in globals.binclf2name.items():
      clf = model2binclfs[md][cutoff]
      dataset.ytrain = dpp.gen_y_nnt_binary(ytrain, cutoff)
      dataset.ytest = dpp.gen_y_nnt_binary(ytest, cutoff)
      # Evaluate model
      pred_train, pred_test, ix = model_eval.eval_binary_clf(clf, cutoff, dataset, md_name,
                                                             metric=metric, plot_roc=False)
      # # Generate feature importance
      # model_eval.gen_feature_importance_bin_clf(clf, md, Xtest, dataset.ytest, cutoff=cutoff)

      # Plot ROC curve for the model at 'cutoff'
      plot_roc_curve(clf, Xtrain, dataset.ytrain, name=md_name, ax=axs[0])
      plot_roc_curve(clf, Xtest, dataset.ytest, name=md_name, ax=axs[1])
      if metric == globals.GMEAN:
        pltutil.plot_roc_best_threshold(Xtrain, dataset.ytrain, clf, axs[0])
        pltutil.plot_roc_best_threshold(Xtest, dataset.ytest, clf, axs[1])
      elif metric == globals.FPRPCT15:
        continue

      # Plot calibration curve  TODO: decision_function() or predict_proba() ?
      prob_class1_train, prob_class1_test = clf.predict_proba(Xtrain)[:, 1], clf.predict_proba(Xtest)[:, 1]
      fop_train, mpv_train = calibration_curve(dataset.ytrain, prob_class1_train, n_bins=10, normalize=True)
      fop_test, mpv_test = calibration_curve(dataset.ytest, prob_class1_test, n_bins=10, normalize=True)
      # TODO: add probability distribution hist/bar plot

      axs2[0].plot(mpv_train, fop_train, marker='.', label=md_name)
      axs2[1].plot(mpv_test, fop_test, marker='.', label=md_name)

    pltutil.plot_roc_basics(axs[0], cutoff, 'training')
    pltutil.plot_roc_basics(axs[1], cutoff, 'test')
    pltutil.plot_calibration_basics(axs2[0], cutoff, 'training')
    pltutil.plot_calibration_basics(axs2[1], cutoff, 'test')
    plt.show()

  dataset.ytrain = ytrain
  dataset.ytest = ytest