# Helper functions for model evaluation
from collections import Counter, defaultdict
from dataclasses import dataclass

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from . import globals
from . import utils_plot as pltutil
from . import c1_data_preprocessing as dpp
from .c1_data_preprocessing import Dataset


@dataclass
class ModelPerf:
  """Class for saving evaluated stats of a model"""
  XType: str
  model_name: str
  trained_md: object
  acc: float
  acc_1nnt_tol: float
  acc_2nnt_tol: float
  overpred_pct: float
  underpred_pct: float
  rmse: float
  f1s: pd.DataFrame
  rsqr: float = np.nan
  count: int = np.nan
  count_pct: float = np.nan

  def __str__(self):
    perf_str = "Model: %s\n" \
               "X type: %s\n" \
               "Accuracy: %.2f\n" \
               "Accuracy (tol = 1 NNT): %.2f\n" \
               "Accuracy (tol = 2 NNT): %.2f\n" \
               "Over Prediction Rate: %.2f\n" \
               "Under Prediction Rate: %.2f\n" \
               "RMSE: %.2f\n" \
               "R squared: %.2f\n" \
               "F1 scores: %s\n" \
               "Case Count: %d\n" \
               "Case Count Ratio: %.2f\n" % \
               (self.model_name, self.XType, self.acc, self.acc_1nnt_tol, self.acc_2nnt_tol, self.overpred_pct,
                self.underpred_pct, self.rmse, self.rsqr, self.f1s.to_string(), self.count, self.count_pct)
    return perf_str

  def get_perf_as_dict(self):  # TODO: f1 scores as a list
    return {"Accuracy": self.acc, "Accuracy\n(tol = 1 NNT)": self.acc_1nnt_tol, "Accuracy\n(tol = 2 NNT)": self.acc_2nnt_tol,
            "Over Prediction Rate": self.overpred_pct, "Under Prediction Rate": self.underpred_pct, "RMSE": self.rmse,
            "Case Count": self.count, "Case Count Ratio": self.count_pct}

  @classmethod
  def get_metrics_formatter(cls):
    return {"Accuracy": "{:.2%}".format, "Accuracy\n(tol = 1 NNT)": "{:.2%}".format, "Accuracy\n(tol = 2 NNT)": "{:.2%}".format,
            "Over Prediction Rate": "{:.2%}".format, "Under Prediction Rate": "{:.2%}".format,
            "RMSE": "{:.2f}".format, "Case Count": "{:.0f}".format, "Case Count Ratio": "{:.2%}".format}

  @classmethod
  def get_perf_metrics(cls):
    return ("Accuracy", "Accuracy\n(tol = 1 NNT)", "Accuracy\n(tol = 2 NNT)", "Over Prediction Rate",
            "Under Prediction Rate", "RMSE", "Case Count", "Case Count Ratio")


# @dataclass
# class ModelPerf_XDISAGREE:
#   model_perf: ModelPerf
#   sps_model_perf: ModelPerf
#
#   def get_perf_as_dict(self):  # TODO: f1 scores as a list
#     return {"Accuracy": self.model_perf.acc, "Accuracy (tol = 1 NNT)": self.model_perf.acc_1nnt_tol,
#             "Over Prediction Rate": self.model_perf.overpred_pct, "Under Prediction Rate": self.model_perf.underpred_pct,
#             "RMSE": self.model_perf.rmse, "R-squared": self.model_perf.rsqr,
#             }
#
#   @classmethod
#   def get_metrics_formatter(cls):
#     return {"Accuracy": "{:.2%}".format, "Accuracy (tol = 1 NNT)": "{:.2%}".format,
#             "Over Prediction Rate": "{:.2%}".format, "Under Prediction Rate": "{:.2%}".format,
#             "RMSE": "{:.2f}".format, "R-squared": "{:.2f}".format, "Surgeon Agreement Rate": "{:.2%}".format}
#
#   @classmethod
#   def get_perf_metrics(cls):
#     return ("Accuracy", "Accuracy (tol = 1 NNT)", "Over Prediction Rate", "Under Prediction Rate", "RMSE", "R-squared",
#             "Surgeon Agreement Rate")



@dataclass
class BinaryModelPerf:
  """Class for saving evaluated stats of a binary classifier"""
  XType: str
  model_name: str
  trained_md: object
  task: str
  majority_pct: float
  acc: float
  prec: float
  recall: float
  f1: float

  def __str__(self):
    perf_str = "Model: %s\n" \
               "X type: %s\n" \
               "Task: %s\n" \
               "Majority Percentage: %.2f\n" \
               "Accuracy: %.2f\n" \
               "Precision: %.2f\n" \
               "Recall: %.2f\n" \
               "F1 score: %.2f\n" % \
               (self.model_name, self.XType, self.task, self.majority_pct, self.acc, self.prec, self.recall, self.f1)
    return perf_str

  @classmethod
  def get_metrics_formatter(cls):
    return {"Majority Percentage": "{:.2%}".format, "Accuracy": "{:.2%}".format, "Precision": "{:.2f}".format,
            "Recall": "{:.2f}".format, "F1": "{:.2f}".format}

  @classmethod
  def get_perf_metrics(cls):
    return ("Majority Percentage", "Accuracy", "Precision", "Recall", "F1")


Xtype_to_conf_style = {globals.XTRAIN: sn.color_palette("ch:start=.2,rot=-.3"),
                       globals.XTEST: 'rocket_r'}
Xtype_to_conf_style = defaultdict(lambda: 'rocket_r', Xtype_to_conf_style)


def gen_confusion_matrix(yTrue, yPred, md_name, Xtype, normalize='true', plot=True):
  confmat = metrics.confusion_matrix(yTrue, yPred, labels=np.arange(0, globals.MAX_NNT + 2, 1), normalize=normalize)

  if plot:
    cmap = Xtype_to_conf_style[Xtype]
    title = 'Confusion Matrix (%s - %s)' % (md_name, Xtype)

    # plot confusion matrix
    figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 10), gridspec_kw={'width_ratios': [2, 1]})
    sn.set(font_scale=1.3)  # for label size
    sn.heatmap(confmat, fmt=".2%", cmap=cmap, linecolor='white', linewidths=0.5,
               annot=True, annot_kws={"size": 15}, ax=axs[0])  # font size
    axs[0].set_title(title, fontsize=20, y=1.02)
    axs[0].set_xlabel("Predicted outcome", fontsize=16)
    axs[0].set_ylabel("True outcome", fontsize=16)
    axs[0].set_xticks(np.arange(globals.MAX_NNT+2)+0.5)
    axs[0].set_xticklabels(globals.NNT_CLASS_LABELS, fontsize=13)
    axs[0].set_yticks(np.arange(globals.MAX_NNT+2)+0.5)
    axs[0].set_yticklabels(globals.NNT_CLASS_LABELS, fontsize=13)

    # Plot a vertical histogram of outcome distribution
    outcome_cnter = Counter(yTrue)
    axs[1].barh(range(globals.MAX_NNT+2), [outcome_cnter[i] for i in range(globals.MAX_NNT+2)], align='center')
    axs[1].set_xlabel("Number of surgical cases")
    axs[1].invert_yaxis()
    axs[1].set_yticks(range(globals.MAX_NNT+2))
    axs[1].set_yticklabels(globals.NNT_CLASS_LABELS, fontsize=13)
    rects = axs[1].patches
    total_cnt = len(yTrue)
    labels = ["{:.1%}".format(outcome_cnter[i] / total_cnt) for i in range(globals.MAX_NNT+2)]
    for rect, label in zip(rects, labels):
      ht, wd = rect.get_height(), rect.get_width()
      axs[1].text(wd + 2.5, rect.get_y() + ht / 2, label,
                  ha='left', va='center', fontsize=15)

    figs.tight_layout()
    plt.show()

  return confmat


def eval_nnt_regressor(reg, dataset: Dataset, md_name):
  Xtrain, ytrain, Xtest, ytest = dataset.Xtrain, dataset.ytrain, dataset.Xtest, dataset.ytest

  # predict and round to the nearest int
  pred_train, pred_test = np.rint(reg.predict(Xtrain)), np.rint(reg.predict(Xtest))
  # bucket them into finite number of classes
  pred_train[pred_train > globals.MAX_NNT] = globals.MAX_NNT + 1
  pred_test[pred_test > globals.MAX_NNT] = globals.MAX_NNT + 1

  print("Accuracy (training): ", metrics.accuracy_score(ytrain, pred_train, normalize=True))
  print("Accuracy (test): ", metrics.accuracy_score(ytest, pred_test, normalize=True))

  # Confusion matrix
  labels = [str(i) for i in range(globals.MAX_NNT + 2)]
  labels[-1] = '%s+' % globals.MAX_NNT
  gen_confusion_matrix(ytrain, pred_train, md_name, isTrain=True)
  gen_confusion_matrix(ytest, pred_test, md_name, isTrain=False)

  # Error histogram
  figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
  pltutil.plot_error_histogram(ytrain, pred_train, md_name, Xtype='train', yType="Number of nights", ax=axs[0])
  pltutil.plot_error_histogram(ytest, pred_test, md_name, Xtype='test', yType="Number of nights", ax=axs[1])


def scorer_1nnt_tol(ytrue, ypred):
  # accuracy within +-1 nnt error tolerance
  acc_1nnt_tol = len(np.where(np.abs(ytrue - ypred) <= 1)[0]) / len(ytrue)
  return acc_1nnt_tol


def scorer_2nnt_tol(ytrue, ypred):
  # accuracy within +-1 nnt error tolerance
  acc_1nnt_tol = len(np.where(np.abs(ytrue - ypred) <= 2)[0]) / len(ytrue)
  return acc_1nnt_tol


def scorer_overpred_pct(ytrue, ypred, diff=2):
  overpred_pct = len(np.where((ypred - ytrue) > diff)[0]) / len(ytrue)
  return overpred_pct


def scorer_underpred_pct(ytrue, ypred, diff=2):
  underpred_pct = len(np.where((ytrue - ypred) > diff)[0]) / len(ytrue)
  return underpred_pct


def rmse(g):
  rmse = np.sqrt(metrics.mean_squared_error(g['NUM_OF_NIGHTS'], g['Predicted']))
  return pd.Series(dict(rmse=rmse))


def get_rsqr(clf, md_name, X, ytrue, ypred):
  if ('Ordinal' in md_name) or ('Ensemble' in md_name) or ('Surgeon' in md_name):
    return metrics.r2_score(ytrue, ypred)
  else:
    return clf.score(X, ytrue)


def get_surgeon_agreed_pct(Xdf, ypred_md, population_size):
  if globals.SPS_LOS_FTR not in Xdf.columns:
    return np.nan
  surgeon_pred = dpp.gen_y_nnt(Xdf[globals.SPS_LOS_FTR])
  population_size = len(ypred_md) if population_size == None else population_size
  agree_pct = len(np.where(surgeon_pred == ypred_md)[0]) / population_size
  return agree_pct


def eval_multiclf_on_Xy(clf, Xdf, X, y, md_name, XType, yType=globals.NNT, cohort=globals.COHORT_ALL, pop_size=None,
                        plot=True):
  if md_name == globals.clf2name_eval[globals.SURGEON]:
    pred_y = dpp.gen_y_nnt(Xdf[globals.SPS_LOS_FTR])
  else:
    pred_y = clf.predict(X)
  Xdf = Xdf.copy(deep=True)

  # Numeric Metrics
  mse = metrics.mean_squared_error(y, pred_y)
  rmse = np.sqrt(mse)
  acc = metrics.accuracy_score(y, pred_y, normalize=True)
  acc_1nnt = scorer_1nnt_tol(y, pred_y)
  acc_2nnt = scorer_2nnt_tol(y, pred_y)
  overpred_pct = scorer_overpred_pct(y, pred_y)
  underpred_pct = scorer_underpred_pct(y, pred_y)
  f1s = pd.DataFrame({"NNT class": globals.NNT_CLASS_LABELS,
                      "F-1 score": metrics.f1_score(y, pred_y, labels=globals.NNT_CLASSES, average=None)})
  rsqr = metrics.r2_score(y, pred_y)
  #surgeon_agree_pct = get_surgeon_agreed_pct(Xdf, pred_y, pop_size)
  count_pct = 1.0 if pop_size is None else len(pred_y) / pop_size
  model_perf = ModelPerf(XType, md_name, clf, acc, acc_1nnt, acc_2nnt, overpred_pct, underpred_pct, rmse, f1s, rsqr,
                         len(pred_y), count_pct)

  if plot:
    # Confusion Matrix
    # conf_mat = gen_confusion_matrix(y, pred_y, md_name, XType)

    # Error Histogram
    figs, axs = plt.subplots(nrows=3, ncols=1, figsize=(14, 24))
    pltutil.plot_error_histogram(y, pred_y, md_name, XType, yType=yType, ax=axs[0])
    pltutil.plot_error_histogram(y, pred_y, md_name, XType, yType=yType, ax=axs[1], groupby_outcome=True)
    pltutil.plot_error_hist_pct(y, pred_y, md_name, XType, yType=yType, ax=axs[2])

    # Error distribution among primary procedure
    Xdf['Predicted'] = pred_y
    if 'Error' not in Xdf.columns:
      Xdf['Error'] = pred_y - y
      Xdf['Abs Error'] = np.abs(pred_y - y)
    #display(Xdf[['SURG_CASE_KEY', 'Predicted', 'Error', 'Abs Error']].head(30))
    pproc_df = Xdf.groupby(by=['PRIMARY_PROC']).size().reset_index(name='Counts')
    pproc_df['Count (%)'] = pproc_df['Counts'] / len(pred_y)

    pproc_df = pproc_df\
      .join((Xdf['Predicted'].eq(Xdf[yType])).groupby(Xdf['PRIMARY_PROC'], observed=True).mean()
            .reset_index(name='Accuracy')
            .set_index('PRIMARY_PROC'),
            on='PRIMARY_PROC', how='left')\
      .join(Xdf.groupby(by=['PRIMARY_PROC'], observed=True)['Abs Error'].max()
            .reset_index(name='Max Abs Error')
            .set_index('PRIMARY_PROC'),
            on='PRIMARY_PROC', how='left'
            )\
      .join(Xdf.groupby(by=['PRIMARY_PROC'], observed=True)['Error'].std()
            .reset_index(name='Error Std')
            .set_index('PRIMARY_PROC'),
            on='PRIMARY_PROC', how='left'
            )\
      .join(Xdf.groupby(by=['PRIMARY_PROC'])['Error'].apply(lambda x: x.pow(2).mean()**0.5)
            .reset_index(name='RMSE')
            .set_index('PRIMARY_PROC'),
            on='PRIMARY_PROC', how='left'
            )\
      .join(Xdf.groupby(by=['PRIMARY_PROC'])[yType].std()
            .reset_index(name='True Outcome Std')
            .set_index('PRIMARY_PROC'),
            on='PRIMARY_PROC', how='left')
    pproc_df.sort_values(by=['Counts', 'Accuracy'], ascending=False, inplace=True)
    if cohort != globals.COHORT_ALL:
      pprocs = globals.COHORT_TO_PPROCS[cohort]
      pproc_df = pproc_df.query("PRIMARY_PROC in @pprocs")
      cohort_cnt = pproc_df['Counts'].sum()
      pproc_df.loc[len(pproc_df.index)] = ['Cohort Total', cohort_cnt, 1.0,
                                           (pproc_df['Counts'] * pproc_df['Accuracy']).sum() / cohort_cnt,
                                            pproc_df['Max Abs Error'].max(), np.nan, np.nan, np.nan]

    pproc_df_all = pproc_df.head(30).style\
        .set_table_attributes("style='display:inline'")\
        .set_caption("Model Performance Grouped by Primary Procedure (%s cases)" % XType) \
        .set_properties(**{'text-align': 'center'})\
        .format("{:.2%}", subset=["Count (%)", "Accuracy"])\
        .format("{:.2f}", subset=["Max Abs Error", "Error Std", "RMSE", "True Outcome Std"])
    # if md_name == globals.clf2name[globals.RMFCLF]:
    #   pproc_df.to_csv('primary_procedures_error_cases(rmf-bal).csv')
    #   pproc_df_all.to_csv('pproc_error.csv')
    display(pproc_df_all)

    # Error distribution among CPTs

  return model_perf


# TODO: Replace eval_binary_clf() with this function that yields better format
def eval_binary_clf_on_Xy(clf, cutoff, Xdf, X, y, md_name, XType, yType=globals.NNT, cohort=globals.COHORT_ALL,
                          confmat_ax=None):
  upper_ratio_pct = np.sum(y) / len(y)
  majority_pct = max(1.0 - upper_ratio_pct, upper_ratio_pct)

  pred = clf.predict(X)
  acc = metrics.accuracy_score(y, pred, normalize=True)
  prec = metrics.precision_score(y, pred)
  recall = metrics.recall_score(y, pred)
  f1 = metrics.f1_score(y, pred)
  bin_model_perf = BinaryModelPerf(XType, md_name, clf, "%s <= %d" % (yType, cutoff), majority_pct,
                                   acc, prec, recall, f1)

  # Confusion Matrix
  confmat = metrics.confusion_matrix(y, pred, labels=[0, 1], normalize='true')
  class_names = ['<= ' + str(cutoff), '%d+' % cutoff]

  if confmat_ax is None:
    fig, confmat_ax = plt.subplots(1, 1, figsize=(6, 5))
  sn.heatmap(confmat, fmt=".2%", cmap=Xtype_to_conf_style[XType], square=True,
             annot=True, annot_kws={"size": 18, "weight": "bold"}, ax=confmat_ax, linecolor='white', linewidths=0.8)
  confmat_ax.set_title("Cutoff =%d: Confusion matrix\n(%s - training)" % (cutoff, md_name),
                   y=1.01, fontsize=17)
  confmat_ax.set_xlabel("Predicted Class", fontsize=16)
  confmat_ax.set_ylabel("True Class", fontsize=16)
  confmat_ax.set_xticks(np.arange(2) + 0.5)
  confmat_ax.set_xticklabels(class_names, fontsize=15)
  confmat_ax.set_yticks(np.arange(2) + 0.5)
  confmat_ax.set_yticklabels(['< ' + str(cutoff), '%d+' % cutoff], fontsize=15)

  return bin_model_perf


def eval_binary_clf(clf, cutoff, dataset: Dataset, md_name, plot_calib_curve=True, plot_roc=True,
                    metric=None, axs=None):
  """If built-in pred = False, optimize for geometric mean of trp & 1-fpr"""
  # Naive guess acc
  Xtrain, ytrain, Xtest, ytest = dataset.Xtrain, dataset.ytrain, dataset.Xtest, dataset.ytest
  upper_ratio_train, upper_ratio_test = np.sum(ytrain) / len(ytrain), np.sum(ytest) / len(ytest)
  print("\n%s:" % md_name)
  print("Naive guess accuracy (training): ", max(1.0 - upper_ratio_train, upper_ratio_train))
  print("Naive guess accuracy (test): ", max(1.0 - upper_ratio_test, upper_ratio_test))

  if metric is None:
    pred_train, pred_val = clf.predict(Xtrain), clf.predict(Xtest)
    ix = None
  else:
    pred_train_prob, pred_val_prob = clf.predict_proba(Xtrain)[:, 1], clf.predict_proba(Xtest)[:, 1]  # focus on class1 only
    if metric == globals.GMEAN:
      # select an optimal threshold based on geometric mean / Youden’s J statistic of tpr and (1-fpr)
      fpr, tpr, thresholds = metrics.roc_curve(ytrain, pred_train_prob)
      ix = np.argmax(tpr - fpr)
    elif metric == globals.F1:
      # select threshold based on F1 score
      prec, recall, thresholds = metrics.precision_recall_curve(ytrain, pred_train_prob)
      ix = np.argmax((2 * prec * recall) / (prec + recall))
    else:
      raise NotImplementedError("Metric %s is not implemented yet!" % metric)
    # Update prediction based on the selected optimal threshold
    best_threshold = thresholds[ix]
    pred_train, pred_val = np.zeros_like(ytrain), np.zeros_like(ytest)
    pred_train[pred_train_prob > best_threshold] = 1
    pred_val[pred_val_prob > best_threshold] = 1

  # Acc, precision, sensitivity, F-1 score
  print("Accuracy (training): ", metrics.accuracy_score(ytrain, pred_train, normalize=True))
  print("Accuracy (test): ", metrics.accuracy_score(ytest, pred_val, normalize=True))
  print("F1 score (training): ", metrics.f1_score(ytrain, pred_train))
  print("F1 score (test): ", metrics.f1_score(ytest, pred_val))

  # Confusion matrix
  confmat_train = metrics.confusion_matrix(ytrain, pred_train, labels=[0, 1], normalize='true')# normalize='true'
  confmat_val = metrics.confusion_matrix(ytest, pred_val, labels=[0, 1], normalize='true') #normalize='true'
  class_names = ['<= ' + str(cutoff), '%d+' % cutoff]

  if axs is None:
    figs, axs = plt.subplots(1, 2, figsize=(13, 5)) #fmt=".2%",
  sn.heatmap(confmat_train, fmt=".2%", cmap=sn.color_palette("ch:start=.2,rot=-.3"), square=True,
             annot=True, annot_kws={"size": 18, "weight": "bold"}, ax=axs[0], linecolor='white', linewidths=0.8)
  axs[0].set_title("Cutoff =%d: Confusion matrix\n(%s - training)" % (cutoff, md_name),
                   y=1.01, fontsize=17)
  axs[0].set_xlabel("Predicted Class", fontsize=16)
  axs[0].set_ylabel("True Class", fontsize=16)
  axs[0].set_xticks(np.arange(2) + 0.5)
  axs[0].set_xticklabels(class_names, fontsize=15)
  axs[0].set_yticks(np.arange(2) + 0.5)
  axs[0].set_yticklabels(['< ' + str(cutoff), '%d+' % cutoff], fontsize=15)

  sn.heatmap(confmat_val, fmt=".2%", cmap='rocket_r', square=True, linecolor='white', linewidths=0.8,
             annot=True, annot_kws={"size": 18, "weight": "bold"}, ax=axs[1]) #
  axs[1].set_title("Cutoff =%d: Confusion matrix\n(%s - validation)" % (cutoff, md_name),
                   y=1.01, fontsize=17)
  axs[1].set_xlabel("Predicted Class", fontsize=16)
  axs[1].set_ylabel("True Class", fontsize=16)
  axs[1].set_xticks(np.arange(2) + 0.5)
  axs[1].set_xticklabels(class_names, fontsize=15)
  axs[1].set_yticks(np.arange(2) + 0.5)
  axs[1].set_yticklabels(['< ' + str(cutoff), '%d+' % cutoff], fontsize=15)

  # Calibration curve
  if plot_calib_curve:
    pass

  # ROC curve
  if plot_roc:
    fig, ax = plt.subplots(figsize=(10, 8))
    metrics.plot_roc_curve(clf, Xtest, ytest, ax=ax)
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.set_title("Cutoff=%d: ROC Curve (%s)" % (cutoff, md_name), y=1.01, fontsize=18)
  return pred_train, pred_val, ix


def gen_feature_importance(model, mdabbr, reg=True, ftrs=globals.FEATURE_COLS, pretty_print=False, plot_top_K=15):
  # ftrs must match the data matrix features exactly (same order)
  sorted_frts = [(ftr, imp) for ftr, imp in sorted(zip(ftrs, model.feature_importances_), reverse=True, key=lambda i: i[1])]
  # sorted_frts = [(x, y) for y, x in sorted(zip(model.feature_importances_, ftrs), reverse=True, key=lambda p: p[0])]
  if pretty_print:
    print("\n" + globals.reg2name[mdabbr] if reg else globals.clf2name[mdabbr] + ":")
    c = 1
    for x, y in sorted_frts:
      print("{c}.{ftr}:  {score}".format(c=c, ftr=x, score=round(y, 4)))
      c += 1
  ftr_importance = sorted_frts[:plot_top_K]
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.barh([x[0] for x in ftr_importance], [x[1] for x in ftr_importance])
  ax.invert_yaxis()
  ax.set_xlabel("Feature importance")
  ax.set_ylabel("Features")
  ax.set_title("Top %d most important features (%s)" % (plot_top_K, mdabbr))
  plt.show()
  return sorted_frts


def gen_feature_importance_bin_clf(clf, md, X, y, cutoff=None, ftrs=globals.FEATURE_COLS):
  """
  Generates feature importance of a fitted classifier, clf

  :param clf:
  :param md:
  :return:
  """
  if md == globals.LGR or md == globals.SVC:
    ftr_importance = permutation_importance(clf, X, y, n_repeats=5, random_state=0).importances_mean
  elif md == globals.DTCLF or md == globals.RMFCLF or md == globals.GBCLF or md == globals.XGBCLF:
    ftr_importance = clf.feature_importances_
    #sorted_frts = [(x, y) for y, x in sorted(zip(clf.feature_importances_, ftrs), reverse=True, key=lambda p: p[0])]
  else:
    raise NotImplementedError("%s is not implemented yet!" % md)

  sorted_idx = ftr_importance.argsort()
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.barh(np.array(ftrs)[sorted_idx], ftr_importance[sorted_idx])
  if cutoff is not None:
    ax.set_xlabel("Cutoff=%d: Feature Importance (%s)" % (cutoff, globals.clf2name[md]), fontsize=15)
  else:
    ax.set_xlabel("Feature Importance (%s)" % globals.clf2name[md], fontsize=15)


def eval_model_by_cpts():
  # TODO 1. First split by number of CPTs, then evaluate by the tuples within the set with same #cpts
  # TODO 2. Aggregate everything and evaluate simply by CPT
  pass


def eval_model_by_primary_proc():

  pass


def eval_model_by_ccsrs():
  pass

