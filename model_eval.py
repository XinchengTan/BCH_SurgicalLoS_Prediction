# Helper functions for model evaluation
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn import metrics
from sklearn.inspection import permutation_importance

from . import globals


def gen_error_histogram(true_y, pred_y, md_name, Xtype='train', yType="LoS (days)", ax=None):
  error = np.array(pred_y) - np.array(true_y)
  bins = np.arange(-globals.MAX_NNT, globals.MAX_NNT+1, 1)

  if ax is None:
    plt.hist(error, bins=bins, align='left', color='grey')
    plt.title("Prediction Error Histogram (%s - %s)" % (md_name, Xtype), y=1.01, fontsize=18)
    plt.xlabel(yType, fontsize=16)
    plt.xticks(bins)
    plt.ylabel("Number of surgical cases", fontsize=16)
    plt.show()
  else:
    counts, bins, patches = ax.hist(error, bins=bins, align='left', color='gray')
    ax.set_title("Prediction Error Histogram (%s - %s)" % (md_name, Xtype), y=1.01, fontsize=18)
    ax.set_xlabel(yType, fontsize=16)
    ax.set_xticks(bins)
    ax.set_ylabel("Number of surgical cases", fontsize=16)

    rects = ax.patches
    total_cnt = sum(counts)
    labels = ["{:.1%}".format(cnt / total_cnt) for cnt in counts]
    for rect, label in zip(rects, labels):
      height = rect.get_height()
      ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label,
              ha='center', va='bottom', fontsize=11)
  return error


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
    axs[0].set_xticks(np.arange(9)+0.5)
    axs[0].set_xticklabels([str(i) for i in range(8)] + ['7+'], fontsize=13)
    axs[0].set_yticks(np.arange(9)+0.5)
    axs[0].set_yticklabels([str(i) for i in range(8)] + ['7+'], fontsize=13)

    # Plot a vertical histogram of outcome
    outcome_cnter = Counter(yTrue)
    axs[1].barh(range(9), [outcome_cnter[i] for i in range(9)], align='center')
    axs[1].set_xlabel("Number of surgical cases")
    axs[1].invert_yaxis()
    axs[1].set_yticks(range(9))
    axs[1].set_yticklabels([str(i) for i in range(8)] + ['7+'], fontsize=13)
    figs.tight_layout()
    plt.show()

  return confmat


def eval_nnt_regressor(reg, Xtrain, ytrain, Xval, yval, md_name, Xtest=None, ytest=None):
  # predict and round to the nearest int
  pred_train, pred_test = np.rint(reg.predict(Xtrain)), np.rint(reg.predict(Xval))
  # bucket them into finite number of classes
  pred_train[pred_train > globals.MAX_NNT] = globals.MAX_NNT + 1
  pred_test[pred_test > globals.MAX_NNT] = globals.MAX_NNT + 1

  print("Accuracy (training): ", metrics.accuracy_score(ytrain, pred_train, normalize=True))
  print("Accuracy (validation): ", metrics.accuracy_score(yval, pred_test, normalize=True))

  # Confusion matrix
  labels = [str(i) for i in range(globals.MAX_NNT + 2)]
  labels[-1] = '%s+' % globals.MAX_NNT
  gen_confusion_matrix(ytrain, pred_train, md_name, isTrain=True)
  gen_confusion_matrix(yval, pred_test, md_name, isTrain=False)

  # Error histogram
  figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
  gen_error_histogram(ytrain, pred_train, md_name, Xtype='train', yType="Number of nights",
                                 ax=axs[0])
  gen_error_histogram(yval, pred_test, md_name, Xtype='validation', yType="Number of nights",
                                 ax=axs[1])


def eval_multi_clf(clf, Xtrain, ytrain, Xval, yval, md_name, Xtest=None, ytest=None):
  pred_train, pred_val = clf.predict(Xtrain), clf.predict(Xval)
  train_mse = metrics.mean_squared_error(ytrain, pred_train)
  val_mse = metrics.mean_squared_error(yval, pred_val)
  print("%s:" % md_name)
  print("Accuracy (training): ", metrics.accuracy_score(ytrain, pred_train, normalize=True))
  print("Accuracy (validation): ", metrics.accuracy_score(yval, pred_val, normalize=True))
  class_names = [str(i) for i in range(globals.MAX_NNT+1)] + ["%d+" % globals.MAX_NNT]
  f1_train = pd.DataFrame({"NNT class": class_names, "F-1 score": metrics.f1_score(ytrain, pred_train, average=None)})
  f1_val = pd.DataFrame({"NNT class": class_names, "F-1 score": metrics.f1_score(yval, pred_val, average=None)})
  print("F-1 score (training): ", f1_train)
  print("F-1 score (training): ", f1_val)

  print("R-squared (training set): ", clf.score(Xtrain, ytrain))
  print("R-squared (validation set): ", clf.score(Xval, yval))
  print("MSE (training set): ", train_mse, "RMSE: ", np.sqrt(train_mse))
  print("MSE (validation set): ", val_mse, "RMSE: ", np.sqrt(val_mse))

  # Confusion matrix
  labels = [str(i) for i in range(globals.MAX_NNT + 2)]
  labels[-1] = '%s+' % globals.MAX_NNT
  gen_confusion_matrix(ytrain, pred_train, md_name, isTrain=True)
  gen_confusion_matrix(yval, pred_val, md_name, isTrain=False)

  # Error histogram
  figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
  gen_error_histogram(ytrain, pred_train, md_name, Xtype='train', yType="Number of nights", ax=axs[0])
  gen_error_histogram(yval, pred_val, md_name, Xtype='validation', yType="Number of nights", ax=axs[1])

  return pred_train, pred_val, f1_train, f1_val


def plot_roc_basics(ax, cutoff=None, Xtype='training'):
  ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
  title = "Cutoff=%d: ROC Curve Comparison (%s)" % (cutoff, Xtype) if cutoff is not None else "ROC Curves (%s)" % Xtype
  ax.set_title(title, y=1.01, fontsize=18)
  ax.set_xlabel("False Positive Rate", fontsize=15)
  ax.set_ylabel("True Positive Rate", fontsize=15)
  ax.legend(prop=dict(size=14))


def plot_roc_best_threshold(X, y, clf, ax):
  fpr, tpr, thresholds = metrics.roc_curve(y, clf.predict_proba(X)[:, 1])
  ix = np.argmax(tpr - fpr)
  ax.scatter(fpr[ix], tpr[ix], s=[120], marker='*', color='red')


def plot_roc_fpr_pct_threshold(X, y, clf, ax):
  fpr, tpr, thresholds = metrics.roc_curve(y, clf.predict_proba(X)[:, 1])
  ix = np.argmin()


def eval_binary_clf(clf, cutoff, Xtrain, ytrain, Xval, yval, md_name, plot_roc=True, metric=None, axs=None,
                    Xtest=None, ytest=None):
  """If built-in pred = False, optimize for geometric mean of trp & 1-fpr"""
  # Naive guess acc
  upper_ratio_train, upper_ratio_test = np.sum(ytrain) / len(ytrain), np.sum(yval) / len(yval)
  print("\n%s:" % md_name)
  print("Naive guess accuracy (training): ", max(1.0 - upper_ratio_train, upper_ratio_train))
  print("Naive guess accuracy (validation): ", max(1.0 - upper_ratio_test, upper_ratio_test))

  if metric is None:
    pred_train, pred_val = clf.predict(Xtrain), clf.predict(Xval)
    ix = None
  else:
    pred_train_prob, pred_val_prob = clf.predict_proba(Xtrain)[:, 1], clf.predict_proba(Xval)[:, 1]  # focus on class1 only
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
    pred_train, pred_val = np.zeros_like(ytrain), np.zeros_like(yval)
    pred_train[pred_train_prob > best_threshold] = 1
    pred_val[pred_val_prob > best_threshold] = 1

  # Acc, precision, sensitivity, F-1 score
  print("Accuracy (training): ", metrics.accuracy_score(ytrain, pred_train, normalize=True))
  print("Accuracy (validation): ", metrics.accuracy_score(yval, pred_val, normalize=True))
  print("F1 score (training): ", metrics.f1_score(ytrain, pred_train))
  print("F1 score (validation): ", metrics.f1_score(yval, pred_val))

  # Confusion matrix
  confmat_train = metrics.confusion_matrix(ytrain, pred_train, labels=[0, 1], normalize='true')
  confmat_val = metrics.confusion_matrix(yval, pred_val, labels=[0, 1], normalize='true')
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
  #plt.show()

  # ROC curve
  if plot_roc:
    fig, ax = plt.subplots(figsize=(10, 8))
    metrics.plot_roc_curve(clf, Xval, yval, ax=ax)
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
