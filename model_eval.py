# Helper functions for model evaluation
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from . import globals
from . import plot_utils as pltutil
from .data_preprocessing import Dataset



def gen_confusion_matrix(yTrue, yPred, md_name, isTrain=True, isTest=False, plot=True):
  confmat = metrics.confusion_matrix(yTrue, yPred, labels=np.arange(0, globals.MAX_NNT + 2, 1), normalize='true')

  if plot:
    if isTrain:
      cmap = sn.color_palette("ch:start=.2,rot=-.3")
      title = "Confusion Matrix (%s - training)" % md_name
    else:
      cmap = 'rocket_r'
      title = "Confusion Matrix (%s - test)" % md_name if isTest else "Confusion Matrix (%s - validation)" % md_name

    # plot confusion matrix
    figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 10), gridspec_kw={'width_ratios': [2, 1]})
    sn.set(font_scale=1.3)  # for label size
    sn.heatmap(confmat, fmt=".2%", cmap=cmap, linecolor='white', linewidths=0.5,
               annot=True, annot_kws={"size": 15}, ax=axs[0])  # font size
    axs[0].set_title(title, fontsize=20, y=1.01)
    axs[0].set_xlabel("Predicted outcome", fontsize=16)
    axs[0].set_ylabel("True outcome", fontsize=16)
    axs[0].set_xticks(np.arange(globals.MAX_NNT+2)+0.5)
    axs[0].set_xticklabels([str(i) for i in range(globals.MAX_NNT+1)] + ['7+'], fontsize=13)
    axs[0].set_yticks(np.arange(globals.MAX_NNT+2)+0.5)
    axs[0].set_yticklabels([str(i) for i in range(globals.MAX_NNT+1)] + ['7+'], fontsize=13)

    # Plot a vertical histogram of outcome distribution
    outcome_cnter = Counter(yTrue)
    axs[1].barh(range(globals.MAX_NNT+2), [outcome_cnter[i] for i in range(globals.MAX_NNT+2)], align='center')
    axs[1].set_xlabel("Number of surgical cases")
    axs[1].invert_yaxis()
    axs[1].set_yticks(range(globals.MAX_NNT+2))
    axs[1].set_yticklabels([str(i) for i in range(globals.MAX_NNT+1)] + ['7+'], fontsize=13)
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


def eval_multi_clf(clf, dataset: Dataset, md_name):
  Xtrain, ytrain, Xtest, ytest = dataset.Xtrain, dataset.ytrain, dataset.Xtest, dataset.ytest

  pred_train, pred_val = clf.predict(Xtrain), clf.predict(Xtest)
  train_mse = metrics.mean_squared_error(ytrain, pred_train)
  val_mse = metrics.mean_squared_error(ytest, pred_val)
  print("%s:" % md_name)
  print("Accuracy (training): ", metrics.accuracy_score(ytrain, pred_train, normalize=True))
  print("Accuracy (test): ", metrics.accuracy_score(ytest, pred_val, normalize=True))
  class_names = [str(i) for i in range(globals.MAX_NNT+1)] + ["%d+" % globals.MAX_NNT]
  f1_train = pd.DataFrame({"NNT class": class_names, "F-1 score": metrics.f1_score(ytrain, pred_train, average=None)})
  f1_val = pd.DataFrame({"NNT class": class_names, "F-1 score": metrics.f1_score(ytest, pred_val, average=None)})
  print("F-1 score (training): ", f1_train)
  print("F-1 score (training): ", f1_val)

  if "Ordinal" not in md_name:
    print("R-squared (training set): ", clf.score(Xtrain, ytrain))
    print("R-squared (test set): ", clf.score(Xtest, ytest))
    print("MSE (training set): ", train_mse, "RMSE: ", np.sqrt(train_mse))
    print("MSE (test set): ", val_mse, "RMSE: ", np.sqrt(val_mse))

  # Confusion matrix
  labels = [str(i) for i in range(globals.MAX_NNT + 2)]
  labels[-1] = '%s+' % globals.MAX_NNT
  gen_confusion_matrix(ytrain, pred_train, md_name, isTrain=True)
  gen_confusion_matrix(ytest, pred_val, md_name, isTrain=False)

  # Error histogram
  type_tr, type_tst, yType = 'train', 'test', 'Number of nights'
  figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
  pltutil.plot_error_histogram(ytrain, pred_train, md_name, Xtype=type_tr, yType=yType, ax=axs[0])
  pltutil.plot_error_histogram(ytest, pred_val, md_name, Xtype=type_tst, yType=yType, ax=axs[1])

  figs2, axs2 = plt.subplots(nrows=2, ncols=1, figsize=(18, 20))
  pltutil.plot_error_histogram(ytrain, pred_train, md_name, Xtype=type_tr, yType=yType, ax=axs2[0], groupby_outcome=True)
  pltutil.plot_error_hist_pct(ytrain, pred_train, md_name, Xtype=type_tr, yType=yType, ax=axs2[1])

  figs3, axs3 = plt.subplots(nrows=2, ncols=1, figsize=(18, 20))
  pltutil.plot_error_histogram(ytest, pred_val, md_name, Xtype=type_tst, yType=yType, ax=axs3[0], groupby_outcome=True)
  pltutil.plot_error_hist_pct(ytest, pred_val, md_name, Xtype=type_tst, yType=yType, ax=axs3[1])

  return pred_train, pred_val, f1_train, f1_val


def eval_binary_clf(clf, cutoff, dataset: Dataset, md_name, plot_calib_curve=True, plot_roc=True,
                    metric=None, axs=None):
  """If built-in pred = False, optimize for geometric mean of trp & 1-fpr"""
  # Naive guess acc
  Xtrain, ytrain, Xtest, ytest = dataset.Xtrain, dataset.ytrain, dataset.Xtest, dataset.ytest
  upper_ratio_train, upper_ratio_test = np.sum(ytrain) / len(ytrain), np.sum(ytest) / len(ytest)
  print("\n%s:" % md_name)
  print("Naive guess accuracy (training): ", max(1.0 - upper_ratio_train, upper_ratio_train))
  print("Naive guess accuracy (validation): ", max(1.0 - upper_ratio_test, upper_ratio_test))

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
  print("F1 score (test): ", metrics.f1_score(ytrain, pred_train))
  print("F1 score (test): ", metrics.f1_score(ytest, pred_val))

  # Confusion matrix
  confmat_train = metrics.confusion_matrix(ytrain, pred_train, labels=[0, 1], normalize='true')
  confmat_val = metrics.confusion_matrix(ytest, pred_val, labels=[0, 1], normalize='true')
  class_names = ['< ' + str(cutoff), '%d+' % cutoff]

  if axs is None:
    figs, axs = plt.subplots(1, 2, figsize=(13, 5))
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
             annot=True, annot_kws={"size": 18, "weight": "bold"}, ax=axs[1])
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


def gen_feature_importance(model, mdabbr, reg=True, ftrs=globals.FEATURE_COLS, pretty_print=False):
  sorted_frts = [(x, y) for y, x in sorted(zip(model.feature_importances_, ftrs), reverse=True, key=lambda p: p[0])]
  if pretty_print:
    print("\n" + globals.reg2name[mdabbr] if reg else globals.clf2name[mdabbr] + ":")
    c = 1
    for x, y in sorted_frts:
      print("{c}.{ftr}:  {score}".format(c=c, ftr=x, score=round(y, 4)))
      c += 1
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
