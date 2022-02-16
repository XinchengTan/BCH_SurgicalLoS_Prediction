import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import metrics

import c1_data_preprocessing as dpp
import utils_plot as pltutil
import globals, jyp4_model_eval


def surgeon_model_agree_cases_eval(dataset: dpp.Dataset, md2multiclf, md, agree_nnt_diff):

  pass


def surgeon_model_disagree_cases_eval(dataset: dpp.Dataset, md2multiclf, md, disagree_nnt_diff):
  pass


def surgeon_models_both_wrong(dataset: dpp.Dataset, md2multiclf):
  md2err_df = dict()
  err_intersection = set()
  for md in md2multiclf:
    print("\n\n%s:\n" % globals.clf2name[md])
    err_df = gen_surgeon_model_both_wrong_df(dataset, md2multiclf, md)
    if len(err_intersection) == 0:
      err_intersection = set(err_df.index.tolist())
    else:
      err_intersection = err_intersection.intersection(set(err_df.index.tolist()))
  err_intersection_idx = sorted(list(err_intersection))

  print("\n\nNumber of cases that all models failed: %d" % len(err_intersection_idx))
  err_intersection_df = dataset.cohort_df.iloc[err_intersection_idx].copy()
  err_intersection_df['Error'] = err_intersection_df['SPS_PREDICTED_LOS'] - err_intersection_df['NUM_OF_NIGHTS']
  err_intersection_df = err_intersection_df.sort_values(by=['Error'])
  return md2err_df, err_intersection_df


def gen_surgeon_model_agree_df_and_Xydata(dataset: dpp.Dataset, clf, use_test=True):
  return gen_surgeon_model_agree_or_not_df_and_Xydata(dataset, clf, use_test, agree=True)


def gen_surgeon_model_disagree_df_and_Xydata(dataset: dpp.Dataset, clf, use_test=True, diff=None):
  return gen_surgeon_model_agree_or_not_df_and_Xydata(dataset, clf, use_test, agree=False, diff=diff)


def gen_surgeon_model_agree_or_not_df_and_Xydata(dataset: dpp.Dataset, clf, use_test=True, agree=True, diff=None):

  Xdata, ydata, dataset_df_idx = (dataset.Xtest, dataset.ytest, dataset.test_idx) if use_test \
    else (dataset.Xtrain, dataset.ytrain, dataset.train_idx)

  # Filter the selected dataset by keeping only those with SPS prediction
  sps_pred = dataset.cohort_df.iloc[dataset_df_idx]['SPS_PREDICTED_LOS']
  notnull_data_idx = np.where(~sps_pred.isnull())[0]
  if len(notnull_data_idx) < len(dataset_df_idx):
    dataset_df_idx = dataset_df_idx[notnull_data_idx]
    Xdata, ydata = Xdata[notnull_data_idx, :], ydata[notnull_data_idx]
    sps_pred = dataset.cohort_df.iloc[dataset_df_idx]['SPS_PREDICTED_LOS']

  # Fetch surgeon and model prediction
  surgeon_pred = dpp.gen_y_nnt(sps_pred)
  md_pred = clf.predict(Xdata) if clf != None else surgeon_pred  # If clf == None, we are evaluating surgeon prediction

  # Identify all the cases where model prediction and surgeon estimation agree
  if agree:
    ms_agree_data_idx = np.where(surgeon_pred == md_pred)[0]
  else:
    if diff == None:
      ms_agree_data_idx = np.where(surgeon_pred != md_pred)[0]
    elif type(diff) == int:
      ms_agree_data_idx = np.where(np.abs(surgeon_pred - md_pred) == diff)[0]
    elif type(diff) == tuple:  # a tuple of comparison operator and diff value e.g. (">", 2)
      if diff[0] == '>':
        ms_agree_data_idx = np.where(np.abs(surgeon_pred - md_pred) > diff[1])[0]
      elif diff[0] == '>=':
        ms_agree_data_idx = np.where(np.abs(surgeon_pred - md_pred) >= diff[1])[0]
      else:
        raise NotImplementedError()
    else:
      raise NotImplementedError()

  agree_Xdata = Xdata[ms_agree_data_idx, :]  # generate data matrix of all agreement cases
  agree_ydata = ydata[ms_agree_data_idx]

  # Generate dataframe of all agreement cases
  ms_agree_df_index = dataset_df_idx[ms_agree_data_idx]
  agree_df = dataset.cohort_df.iloc[ms_agree_df_index]  # .copy()

  return agree_df, agree_Xdata, agree_ydata


def gen_surgeon_model_both_wrong_df(dataset: dpp.Dataset, md2multiclf, md, disagree_nnt_diff=2):
  """
  Assert the model is already trained
  :param dataset:
  :param md2multiclf:
  :param md:
  :param disagree_nnt_diff: Count as model & surgeon disagree if their prediction differs by at least this value
  :return:
  """
  true_nnt = dataset.ytest
  surgeon_pred = dpp.gen_y_nnt(dataset.cohort_df.iloc[dataset.test_idx]['SPS_PREDICTED_LOS'])
  clf = md2multiclf[md]
  md_pred = clf.predict(dataset.Xtest)

  # get cases where model prediction and surgeon estimation agree
  ms_agree_test_idx = np.where(surgeon_pred == md_pred)[0]
  ms_agree_df_index = dataset.test_idx[ms_agree_test_idx]

  # Hit Rate of the cohort
  print("Model & Surgeon Hit Rate: %.2f%%" %
        (100 * metrics.accuracy_score(true_nnt[ms_agree_test_idx], md_pred[ms_agree_test_idx])))

  # Percentage of mode-surgeon agreements
  print("\nSurgeon estimation and model prediction agrees in {:.2f}% among {x} cases".format(
    100 * len(ms_agree_test_idx) / len(true_nnt), x=len(true_nnt)))

  # When model and surgeon agrees, what's the percentage of the cases where both are wrong?
  agreed_pred = md_pred[ms_agree_test_idx]
  agree_true = true_nnt[ms_agree_test_idx]
  err = agreed_pred - agree_true
  wrong_by_ge_d_nnts_idx = np.where(np.abs(err) >= disagree_nnt_diff)[0]
  err_ratio = len(wrong_by_ge_d_nnts_idx) / len(agree_true)
  wrong_by_ge_d_nnts_df_index = ms_agree_df_index[wrong_by_ge_d_nnts_idx]

  print("\nIn %d cases when surgeon and model agree, prediction is wrong by >= 2 NNTs for %d cases (%.2f%%)" % (
    len(agree_true), len(wrong_by_ge_d_nnts_idx), 100 * err_ratio))
  ret = dataset.cohort_df.iloc[wrong_by_ge_d_nnts_df_index].copy()
  ret['Error'] = err[wrong_by_ge_d_nnts_idx]
  ret = ret.sort_values(by=['Error'])

  return ret


def eval_surgeon_perf(df, idxs=None, Xtype='all'):
  if idxs is not None:
    df = df.iloc[idxs]
    print(df.shape[0])

  # Evaluate SPS surgeon estimation performance
  true_nnt = dpp.gen_y_nnt(df['NUM_OF_NIGHTS'])
  surgeon_pred = dpp.gen_y_nnt(df['SPS_PREDICTED_LOS'])
  jyp4_model_eval.gen_confusion_matrix(true_nnt, surgeon_pred, globals.SURGEON, Xtype=Xtype)

  print(globals.SURGEON)
  print("Accuracy (%s): " % Xtype, metrics.accuracy_score(true_nnt, surgeon_pred, normalize=True))
  class_names = [str(i) for i in range(globals.MAX_NNT+1)] + ["%d+" % globals.MAX_NNT]
  f1_sps = pd.DataFrame({"NNT class": class_names, "F-1 score": metrics.f1_score(true_nnt, surgeon_pred, average=None)})
  rmse = np.sqrt(metrics.mean_squared_error(true_nnt, surgeon_pred))
  print("F-1 score (%s): " % Xtype, f1_sps)
  print("RMSE (%s)" % Xtype, rmse)

  figs, axs = plt.subplots(nrows=2, ncols=1, figsize=(18, 20))
  pltutil.plot_error_histogram(true_nnt, surgeon_pred, globals.SURGEON, Xtype=Xtype, yType="Number of nights", ax=axs[0],
                                 groupby_outcome=False)
  pltutil.plot_error_hist_pct(true_nnt, surgeon_pred, globals.SURGEON, Xtype=Xtype, yType="Number of nights", ax=axs[1])

  # Binary cases
  figs, axs = plt.subplots(nrows=2, ncols=4, figsize=(21, 10))
  for cutoff in globals.NNT_CUTOFFS:
    true_nnt_b = dpp.gen_y_nnt_binary(true_nnt, cutoff)
    surgeon_pred_b = dpp.gen_y_nnt_binary(surgeon_pred, cutoff)
    # Confusion matrix
    print("\nCutoff = %d" % cutoff)
    print("Accuracy (%s): " % Xtype, metrics.accuracy_score(true_nnt_b, surgeon_pred_b, normalize=True))
    print("F1 score (%s): " % Xtype, metrics.f1_score(true_nnt_b, surgeon_pred_b))
    confmat = metrics.confusion_matrix(true_nnt_b, surgeon_pred_b, labels=[0, 1], normalize='true')
    class_names = ['< ' + str(cutoff), '%d+' % cutoff]

    ax = axs[(cutoff-1) // 4][(cutoff-1) % 4]
    sn.heatmap(confmat, fmt=".2%", cmap=sn.color_palette("ch:start=.2,rot=-.3") if Xtype == 'training' else 'rocket_r',
               annot=True, annot_kws={"size": 16}, ax=ax, linecolor='white', linewidths=0.8)
    ax.set_title("Cutoff =%d: Confusion matrix\n(%s - %s)" % (cutoff, globals.SURGEON, Xtype),
                 y=1.01, fontsize=16)
    ax.set_xlabel("Predicted Class", fontsize=15)
    ax.set_ylabel("True Class", fontsize=15)
    ax.set_xticks(np.arange(2) + 0.5)
    ax.set_xticklabels(class_names, fontsize=14)
    ax.set_yticks(np.arange(2) + 0.5)
    ax.set_yticklabels(['< ' + str(cutoff), '%d+' % cutoff], fontsize=14)
  figs.tight_layout()
  figs.savefig("Surgeon binclf.png")
  return f1_sps
