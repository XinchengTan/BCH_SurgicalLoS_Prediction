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
from collections import defaultdict, Counter

from imblearn.over_sampling import SMOTENC
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, make_scorer, plot_roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from statsmodels.miscmodels.ordinal_model import OrderedModel
from xgboost import XGBClassifier

from . import globals, c4_model_eval, c6_surgeon, utils
from . import c1_data_preprocessing as dpp
from .c1_data_preprocessing import Dataset
from .c4_model_eval import ModelPerf
from .c3_ensemble import Ensemble
from .c8_models import OrdinalClassifier
from . import utils_plot as pltutil


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
    clf = SVC(gamma='auto', class_weight=cls_weight, probability=True).fit(Xtrain, ytrain)  # TODO: consider probability=False to speed up
  elif model == globals.KNN:
    clf = KNeighborsClassifier(n_neighbors=10).fit(Xtrain, ytrain)
  elif model == globals.DTCLF:
    clf = DecisionTreeClassifier(random_state=0, max_depth=4, class_weight=cls_weight).fit(Xtrain, ytrain)
  elif model == globals.RMFCLF:
    max_ftr = 130 if Xtrain.shape[1] > 130 else None
    clf = RandomForestClassifier(max_features=max_ftr, random_state=0, class_weight=cls_weight).fit(Xtrain, ytrain)
  elif model == globals.GBCLF:
    clf = GradientBoostingClassifier(random_state=0, max_depth=2).fit(Xtrain, ytrain)  # TODO: imbalanced class issue here!
  elif model == globals.XGBCLF:
    clf = XGBClassifier(random_state=0).fit(Xtrain, ytrain)
  elif model == globals.ORDCLF_LOGIT:
    clf = OrdinalClassifier(distr='logit', solver='bfgs', disp=True).fit(Xtrain, ytrain)
  elif model == globals.ORDCLF_PROBIT:
    clf = OrdinalClassifier(distr='probit', solver='bfgs', disp=True).fit(Xtrain, ytrain)
  elif model == globals.BAL_BAGCLF:
    clf = BalancedBaggingClassifier(random_state=globals.SEED).fit(Xtrain, ytrain)
  else:
    raise NotImplementedError("Model %s is not supported!" % model)

  if calibrate_method is not None:
    if calibrate_on_val:
      clf = CalibratedClassifierCV(clf, method=calibrate_method, cv='prefit').fit(Xval, yval)
    else:
      clf = CalibratedClassifierCV(clf, method=calibrate_method, cv='prefit').fit(Xtrain, ytrain)

  return clf


def minority_class_size(y):
  return min(Counter(y).values())


def run_classifier_cv(X, y, md, scorer, class_weight=None, kfold=5):
  n_frts = X.shape[1]
  minority_size = minority_class_size(y)
  if md == globals.SVC:
    clf = SVC(random_state=globals.SEED, class_weight=class_weight, probability=False)  # if probability=True, will be slow
    param_space = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                   'gamma': sorted([1 / n_frts] + [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]),
                   'kernel': ['rbf', 'poly', 'sigmoid'],
                   'degree': [2, 3, 4]}
  elif md == globals.KNN:
    clf = KNeighborsClassifier()
    param_space = {
      'algorithm': ['ball_tree', 'kd_tree', 'brute'],
      'leaf_size': list(range(20, minority_size, 10)),
      'metric': ['minkowski'],
      'n_neighbors': list(range(5, minority_size + 1, minority_size//10 + 1)),
      'p': [1, 2, 3],
      'weights': ['uniform', 'distance']
    }
  elif md == globals.DTCLF:
    # 'ccp_alpha': 0.0
    # 'min_impurity_decrease': 0.0,
    # 'min_impurity_split': None,
    # 'criterion': ['gini', 'entropy'],
    clf = DecisionTreeClassifier(random_state=globals.SEED, class_weight=class_weight)
    param_space = {
      'max_depth': [None] + list(range(2, 21, 2)),
      'max_features': list(range(2, 1 + n_frts // 2, 10)) + [n_frts],
      'max_leaf_nodes': [None] + list(range(5, 101, 5)),
      'max_samples': np.arange(0.1, 1, 0.1),
      'min_samples_leaf': [1] + [i for i in range(2, 17, 2)],
      'min_samples_split': list(range(2, int(minority_size * (1 - 1. / kfold)), 3)),
      'splitter': ['best', 'random']
    }
  elif md == globals.RMFCLF:
    # 'oob_score': False
    # 'min_impurity_decrease': 0.0,
    # 'min_impurity_split': None,
    # 'criterion': ['gini', 'entropy'],
    # 'bootstrap': True
    clf = RandomForestClassifier(random_state=globals.SEED, class_weight=class_weight)
    param_space = {
      'max_depth': [None] + list(range(2, 21, 2)),
      'max_features': sorted(set(list(range(2, n_frts // 2 + 1, n_frts//20 + 1)) + [int(np.sqrt(n_frts))])),
      'max_leaf_nodes': [None] + list(range(5, 101, 5)),
      'max_samples': np.arange(0.1, 1, 0.1),
      'min_samples_leaf': [1] + [i for i in range(2, 17, 2)],
      'min_samples_split': list(range(2, int(minority_size * (1 - 1. / kfold)), 3)),
      'n_estimators': [20, 30, 40, 50, 80, 100, 200, 300, 400]
    }
    if not param_space['min_samples_split']:
      raise ValueError("min_samples_split cannot < 2!")
  elif md == globals.GBCLF:
    # 'min_impurity_decrease': 0.0,
    # 'min_impurity_split': None,
    # 'min_weight_fraction_leaf': 0.0,
    # 'ccp_alpha': 0.0
    clf = GradientBoostingClassifier(random_state=globals.SEED, class_weight=class_weight,
                                     validation_fraction=0.15, n_iter_no_change=3)
    param_space = {
      'learning_rate': [0.001, 0.03, 0.01, 0.3, 0.1, 0.3],
      'loss': 'deviance',
      'max_depth': [None] + list(range(2, 21, 2)),
      'max_features': [None] + list(range(2, 1 + n_frts // 2, 2)),
      'max_leaf_nodes': [None] + list(range(5, )),
      'min_samples_leaf': [1, 2, 3, 4, 5],
      'min_samples_split': list(range(2, int(minority_size * (1 - 1. / kfold)), 3)),
      'n_estimators': [20, 30, 40, 50, 80, 100, 200, 300, 400]
    }
    if not param_space['min_samples_split']:
      raise ValueError("min_samples_split cannot < 2!")
  else:
    raise NotImplementedError("Model %s is not supported!" % md)

  # Define scorer
  refit = True
  if scorer == globals.SCR_ACC_ERR1:
    scorer = make_scorer(c4_model_eval.MyScorer.scorer_1nnt_tol, greater_is_better=True)
  elif scorer == globals.SCR_1NNT_TOL_ACC:
    scorer = {globals.SCR_ACC: globals.SCR_ACC,
              globals.SCR_ACC_ERR1: make_scorer(c4_model_eval.MyScorer.scorer_1nnt_tol, greater_is_better=True)}
    refit = globals.SCR_ACC_ERR1
  elif scorer == globals.SCR_MULTI_ALL:
    scorer = {globals.SCR_ACC: globals.SCR_ACC, globals.SCR_AUC: globals.SCR_AUC,
              globals.SCR_ACC_ERR1: make_scorer(c4_model_eval.MyScorer.scorer_1nnt_tol, greater_is_better=True)}
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
      c4_model_eval.eval_nnt_regressor(reg, dataset, md_name)
  else:
    reg = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=model)
    model2trained[model] = reg
    c4_model_eval.eval_nnt_regressor(reg, dataset, md_name=globals.reg2name[model])

  return model2trained


def predict_nnt_multi_clf(dataset: Dataset, model=None, cls_weight=None, evaluate=True, smote=False):
  """ Predict number of nights via a multi-class classfier"""
  Xtrain, ytrain, Xtest, ytest = dataset.Xtrain, dataset.ytrain, dataset.Xtest, dataset.ytest
  if smote:
    Xtrain, ytrain = dpp.gen_smote_Xy(Xtrain, ytrain, dataset.feature_names)

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
    model2ModelPerf, eval_dfs, eval_df_styler = performance_eval_multiclfs(dataset, model2trained, XType=None)
    for eval_df in eval_dfs:
      display(eval_df)

  return model2trained


def performance_eval_multiclfs(dataset: Dataset, model2trained_clf, XType, cohort=globals.COHORT_ALL, eval_surgeon=False,
                               plot=True):
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
  model2trained_clf = dict(model2trained_clf)
  Xtrain, ytrain, Xtest, ytest = dataset.Xtrain, dataset.ytrain, dataset.Xtest, dataset.ytest
  df = dataset.cohort_df
  md2ModelPerf = defaultdict(lambda: defaultdict(lambda: None))
  xtypes = [XType] if XType is not None else [globals.XTRAIN, globals.XTEST]

  # A majority-vote based Ensemble model that equally weighs each clf and outputs the majority prediction
  ensemble_md = Ensemble([globals.MULTI_CLF], utils.gen_md2AllClfs(md2multiclf=model2trained_clf))
  model2trained_clf[globals.ENSEMBLE_MAJ_EQ] = ensemble_md

  xtype2diff = {globals.XDISAGREE: None, globals.XDISAGREE1: 1, globals.XDISAGREE2: 2,
                globals.XDISAGREE_GT2: ('>', 2)}

  # Treat surgeon as a model and evaluate
  if eval_surgeon:
    df_sps = df.join(dataset.df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='inner', rsuffix='_full')
    if not df_sps[globals.SPS_PRED].isnull().any():  # or (XType not in [globals.XTRAIN, globals.XTEST, None])
      df = df_sps
      model2trained_clf[globals.SURGEON] = None
    else:
      warnings.warn("Skipping surgeon prediction evaluation, since some cases have missing SPS prediction")
      #raise Warning("Skipping surgeon prediction evaluation, since some cases have missing SPS prediction")

  md2SurgeonPerf_disagree = defaultdict(lambda: defaultdict(lambda: None))
  for md, clf in model2trained_clf.items():

    if XType is None:  # evaluate both training and test set
      md2ModelPerf[md][globals.XTRAIN] = c4_model_eval.eval_multiclf_on_Xy(clf, dataset.train_cohort_df, Xtrain, ytrain,
                                                                           globals.clf2name_eval[md], globals.XTRAIN,
                                                                           cohort=cohort, plot=plot)
      md2ModelPerf[md][globals.XTEST] = c4_model_eval.eval_multiclf_on_Xy(clf, dataset.test_cohort_df, Xtest, ytest,
                                                                          globals.clf2name_eval[md], globals.XTEST,
                                                                          cohort=cohort, plot=plot)

    elif XType == globals.XTRAIN:
      md2ModelPerf[md][XType] = c4_model_eval.eval_multiclf_on_Xy(clf, dataset.train_cohort_df, Xtrain, ytrain,
                                                                  globals.clf2name_eval[md], XType, cohort=cohort, plot=plot)

    elif XType == globals.XTEST:
      md2ModelPerf[md][XType] = c4_model_eval.eval_multiclf_on_Xy(clf, dataset.test_cohort_df, Xtest, ytest,
                                                                  globals.clf2name_eval[md], XType, cohort=cohort, plot=plot)

    elif XType == globals.XAGREE:
      N = dataset.Xtest.shape[0]
      if md != globals.SURGEON:
        data_df, Xdata, ydata = c6_surgeon.gen_surgeon_model_agree_df_and_Xydata(dataset, clf, use_test=True)
      else:
        data_df, Xdata, ydata = dataset.test_cohort_df, Xtest, ytest
      md2ModelPerf[md][XType] = c4_model_eval.eval_multiclf_on_Xy(
        clf, data_df, Xdata, ydata, globals.clf2name_eval[md], XType, cohort=cohort, pop_size=N, plot=plot)

    else:  # focus on the sample where surgeon and model disagree
      if XType in xtype2diff.keys():
        N = dataset.Xtest.shape[0]
        if md != globals.SURGEON:
          data_df, Xdata, ydata = c6_surgeon.gen_surgeon_model_disagree_df_and_Xydata(
            dataset, clf, use_test=True, diff=xtype2diff[XType])
          md2ModelPerf[md][XType] = c4_model_eval.eval_multiclf_on_Xy(
            clf, data_df, Xdata, ydata, globals.clf2name_eval[md], XType, cohort=cohort, pop_size=N, plot=plot)
          md2SurgeonPerf_disagree[md][XType] = c4_model_eval.eval_multiclf_on_Xy(
            None, data_df, Xdata, ydata, globals.clf2name_eval[globals.SURGEON], XType, cohort=cohort, pop_size=N, plot=plot)
      else:
        raise NotImplementedError("XType '%s' is not implemented yet!" % XType)

  # Remove surgeon model if we are looking at cases where model and surgeon disagree
  if globals.SURGEON in md2ModelPerf.keys() and XType == globals.XDISAGREE:
    md2ModelPerf.pop(globals.SURGEON)

  # Create a dataframe table to present model performance along all the evaluation metrics
  eval_dfs, eval_df_stylers = [], []
  for xtype in xtypes:
    # create a data table from a dict
    if not xtype in xtype2diff.keys():
      md2MP_xtype_dict = {globals.clf2name_eval.get(md, md): md2ModelPerf[md][xtype].get_perf_as_dict() for md in md2ModelPerf}
      df = pd.DataFrame.from_dict(md2MP_xtype_dict, orient='index').sort_values(by=["Accuracy"], ascending=False)
      df.index.name = 'Model'
      styler = df.style \
        .set_table_attributes("style='display:inline'") \
        .set_caption("Model Performance (%s cases)" % xtype) \
        .format(ModelPerf.get_metrics_formatter()) \
        .set_properties(**{'text-align': 'center',
                           'white-space': 'pre-wrap'})

    else:
      md2MP_xtype_dict = dict(sorted({(globals.clf2name_eval.get(md, md), 'Model'): md2ModelPerf[md][xtype].get_perf_as_dict() for md in
                          md2ModelPerf}.items(), key=lambda item: item[1]['Accuracy'], reverse=True))
      md2SP_xtype_dict = {(globals.clf2name_eval.get(md, md), 'Surgeon'): md2SurgeonPerf_disagree[md][xtype].get_perf_as_dict() for md in
                          md2SurgeonPerf_disagree}
      md2Perf_xtype_dict = {**md2MP_xtype_dict, **md2SP_xtype_dict}
      df = pd.DataFrame.from_dict(md2Perf_xtype_dict, orient='index')
      df.index.names = ['Model', 'Predictor']
      df = df.sort_values(by=["Accuracy"], ascending=False).sort_index(level='Model', sort_remaining=False)

      styler = df.style \
        .set_table_attributes("style='display:inline'") \
        .set_caption("Model Performance (%s cases)" % xtype) \
        .format(ModelPerf.get_metrics_formatter()) \
        .set_properties(**{'text-align': 'center',
                           'white-space': 'pre-wrap'})
      # s = df.style
      # for idx, group_df in df.groupby('Model'):
      #   s.set_table_styles({group_df.index[0]: [{'selector': '', 'props': 'border-top: 3px solid black;'}]},
      #                      overwrite=False, axis=1)
      # display(s)
    eval_dfs.append(df)
    eval_df_stylers.append(styler)

  return md2ModelPerf, eval_dfs, eval_df_stylers


def performance_eval_multiclfs_cv(kfold_datasets):
  eval_dfs_all_test, eval_dfs_agree, eval_dfs_disagree = [], [], []
  eval_dfs_disagree1, eval_dfs_disagree2, eval_dfs_disagree_gt2 = [], [], []

  for k in range(len(kfold_datasets)):
    dataset = kfold_datasets[k]
    print("Training Data shape: ", dataset.train_cohort_df.shape)

    md2multiclf = predict_nnt_multi_clf(dataset, model=None, cls_weight=None, evaluate=False)
    md2ModelPerf_test, eval_dfs_test, _ = performance_eval_multiclfs(dataset, md2multiclf, XType=globals.XTEST,
                                                                     eval_surgeon=True, plot=False)
    eval_dfs_all_test.append(eval_dfs_test[0])

    md2ModelPerf_agree, eval_dfs_agree, _ = performance_eval_multiclfs(dataset, md2multiclf, XType=globals.XAGREE,
                                                                     eval_surgeon=True, plot=False)
    eval_dfs_agree.append(eval_dfs_agree[0])

    md2ModelPerf_disagree, eval_dfs_disagree, _ = performance_eval_multiclfs(dataset, md2multiclf, XType=globals.XDISAGREE,
                                                                     eval_surgeon=True, plot=False)
    eval_dfs_disagree.append(eval_dfs_disagree[0])

    md2ModelPerf_disagree1, eval_dfs_disagree1, _ = performance_eval_multiclfs(dataset, md2multiclf,
                                                                             XType=globals.XDISAGREE1,
                                                                             eval_surgeon=True, plot=False)
    eval_dfs_disagree1.append(eval_dfs_disagree1[0])

    md2ModelPerf_disagree2, eval_dfs_disagree2, _ = performance_eval_multiclfs(dataset, md2multiclf,
                                                                             XType=globals.XDISAGREE2,
                                                                             eval_surgeon=True, plot=False)
    eval_dfs_disagree2.append(eval_dfs_disagree2[0])

    md2ModelPerf_disagree_gt2, eval_dfs_disagree_gt2, _ = performance_eval_multiclfs(dataset, md2multiclf,
                                                                             XType=globals.XDISAGREE_GT2,
                                                                             eval_surgeon=True, plot=False)
    eval_dfs_disagree_gt2.append(eval_dfs_disagree_gt2[0])

  # Compile averaged results in one dataframe
  cv_eval_df_all_test = pd.concat(eval_dfs_all_test).groupby(level=0).mean()
  cv_eval_df_agree = pd.concat(eval_dfs_agree).groupby(level=0).mean()
  cv_eval_df_disagree = pd.concat(eval_dfs_disagree).groupby(level=0).mean()
  cv_eval_df_disagree1 = pd.concat(eval_dfs_disagree1).groupby(level=0).mean()
  cv_eval_df_disagree2 = pd.concat(eval_dfs_disagree2).groupby(level=0).mean()
  cv_eval_df_disagree_gt2 = pd.concat(eval_dfs_disagree_gt2).groupby(level=0).mean()


  return cv_eval_df_all_test, cv_eval_df_agree, cv_eval_df_disagree, cv_eval_df_disagree1, cv_eval_df_disagree2, \
         cv_eval_df_disagree_gt2


def predict_nnt_binary_clf(dataset: Dataset, cutoffs, metric=None, model=None, cls_weight=None, eval=True,
                           calibrate_method=None, calibrate_on_val=False, smote=False):
  """
  Predict number of nights via a series of binary classifiers, given a list of integer cutoffs.
  e.g. [1, 2] means two classifiers: one for if NNT < 1 or >= 1, the other for if NNT < 2 or >= 2
  """
  original_Xtrain, Xtest = dataset.Xtrain, dataset.Xtest
  original_ytrain, ytest = np.copy(dataset.ytrain), np.copy(dataset.ytest)

  cutoff2smoteXy = None
  if smote:
    cutoff2smoteXy = {}
    for ct in cutoffs:
      bin_ytrain = dpp.gen_y_nnt_binary(dataset.ytrain, ct)
      ones = sum(bin_ytrain)
      if min(ones, len(bin_ytrain) - ones) < 6:
        cutoff2smoteXy[ct] = None
      else:
        cutoff2smoteXy[ct] = dpp.gen_smote_Xy(dataset.Xtrain, bin_ytrain, dataset.feature_names)

  model2binclfs = {}
  if model is None:
    # fit multiple binary classifiers independently
    # assume classifier_cutoffs is an increasing list of integers, each represent a cutoff value
    for md, md_name in globals.binclf2name.items():
      print("Fitting binary classifiers %s" % md_name)
      cutoff2clf = {}
      for cutoff in cutoffs:
        if smote:
          if cutoff2smoteXy[cutoff] is not None:
            Xtrain, bin_ytrain = cutoff2smoteXy[cutoff]
          else:
            cutoff2clf[cutoff] = None
            continue
        else:
          Xtrain, bin_ytrain = original_Xtrain, dpp.gen_y_nnt_binary(original_ytrain, cutoff)

        # Train classifier
        clf = run_classifier(Xtrain, bin_ytrain, model=md, cls_weight=cls_weight,
                             calibrate_method=calibrate_method, calibrate_on_val=calibrate_on_val)
        cutoff2clf[cutoff] = clf
      model2binclfs[md] = cutoff2clf  # save models

    if eval:
      plot_calib = (calibrate_method != None)
      run_all_eval_binclf(dataset, cutoffs, model2binclfs, metric, plot_calibrate=plot_calib) if not smote \
        else run_all_eval_binclf(dataset, cutoffs, model2binclfs, metric, plot_calibrate=plot_calib,
                                 smoteXytrain=cutoff2smoteXy)

  else:
    # Always evaluate when running each model individually
    cutoff2clf = {}
    figs, axs = plt.subplots(nrows=4, ncols=4, figsize=(21, 21))
    for cutoff in cutoffs:
      dataset.ytrain = dpp.gen_y_nnt_binary(original_ytrain, cutoff)
      dataset.ytest = dpp.gen_y_nnt_binary(ytest, cutoff)
      # Train classifier
      clf = run_classifier(original_Xtrain, dataset.ytrain, model=model, cls_weight=cls_weight, calibrate_method=calibrate_method)
      cutoff2clf[cutoff] = clf

      # Evaluate model
      c4_model_eval.eval_binary_clf(clf, cutoff, dataset, globals.binclf2name[model], metric=metric,
                                    plot_roc=False, axs=[axs[(cutoff-1)//4][(cutoff-1)%4], axs[2+(cutoff-1)//4][(cutoff-1)%4]])

      # Generate feature importance
      #model_eval.gen_feature_importance_bin_clf(clf, model, Xval, yval_b, cutoff=cutoff)
    model2binclfs[model] = cutoff2clf
    figs.tight_layout()
    figs.savefig("%s (val-%s) binclf.png" % (model, str(cls_weight)))
    dataset.ytrain = original_ytrain
    dataset.ytest = ytest
  return model2binclfs


def performance_eval_binclf(dataset: Dataset, model2trained_binclf, XType, cohort=globals.COHORT_ALL):

  return


def run_all_eval_binclf(dataset, clf_cutoffs, model2binclfs, metric=None, plot_calibrate=False, smoteXytrain=None):
  # Input data matrix and response vector
  original_Xtest, original_ytest = dataset.Xtest, np.copy(dataset.ytest)
  original_Xtrain, original_ytrain = np.copy(dataset.Xtrain), np.copy(dataset.ytrain)

  # Actual data matrix and response vector used by the models
  Xtest, ytest = dataset.Xtest, np.copy(dataset.ytest)
  Xtrain, ytrain = dataset.Xtrain, np.copy(dataset.ytrain)

  md2confaxs = {md: plt.subplots(nrows=4, ncols=4, figsize=(21, 21)) for md in globals.binclf2name.keys()}
  for cutoff in clf_cutoffs:
    print("\nCutoff = %d" % cutoff)
    figs, axs = plt.subplots(ncols=2, nrows=1, figsize=(20, 7))  # ROC curves
    if plot_calibrate:
      figs2, axs2 = plt.subplots(ncols=2, nrows=1, figsize=(20, 7))  # Calibration curves
    for md, md_name in globals.binclf2name.items():
      clf = model2binclfs[md][cutoff]
      if clf is None:
        continue
      if smoteXytrain is not None:
        Xtrain, dataset.ytrain = smoteXytrain[cutoff]
        dataset.Xtrain = Xtrain
      else:
        dataset.ytrain = dpp.gen_y_nnt_binary(ytrain, cutoff)
      dataset.ytest = dpp.gen_y_nnt_binary(ytest, cutoff)
      # Evaluate model
      confmat_axs = [md2confaxs[md][1][cutoff//4][cutoff%4], md2confaxs[md][1][2+cutoff//4][cutoff%4]]
      pred_train, pred_test, ix = c4_model_eval.eval_binary_clf(clf, cutoff, dataset, md_name,
                                                                metric=metric, plot_roc=False, axs=confmat_axs)
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
      if plot_calibrate:
        prob_class1_train, prob_class1_test = clf.predict_proba(Xtrain)[:, 1], clf.predict_proba(Xtest)[:, 1]
        fop_train, mpv_train = calibration_curve(dataset.ytrain, prob_class1_train, n_bins=10, normalize=True)
        fop_test, mpv_test = calibration_curve(dataset.ytest, prob_class1_test, n_bins=10, normalize=True)
        # TODO: add probability distribution hist/bar plot
        axs2[0].plot(mpv_train, fop_train, marker='.', label=md_name)
        axs2[1].plot(mpv_test, fop_test, marker='.', label=md_name)

    pltutil.plot_roc_basics(axs[0], cutoff, 'training')
    pltutil.plot_roc_basics(axs[1], cutoff, 'test')
    if plot_calibrate:
      pltutil.plot_calibration_basics(axs2[0], cutoff, 'training')
      pltutil.plot_calibration_basics(axs2[1], cutoff, 'test')
      figs2.show()
    figs.show()

  for md in globals.binclf2name.keys():
    md2confaxs[md][0].tight_layout()
  plt.show()

  # TODO: refactor this into a method in Dataset class
  dataset.Xtrain = original_Xtrain
  dataset.ytrain = original_ytrain
  dataset.ytest = original_ytest
