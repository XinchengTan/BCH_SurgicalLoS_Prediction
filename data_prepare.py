"""
This script contains helper functions to prepare the raw dataset pulled from DB.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from pathlib import Path
from sklearn import metrics

from . import globals
from . import model_eval
from . import data_preprocessing as dpp

pd.set_option('display.max_columns', 50)


def gen_cols_with_nan(df):
  res = df.isnull().sum().sort_values(ascending=False)
  return res


def print_df_info(df, dfname="Dashboard"):
  print("%s columns:\n" % dfname, df.keys())
  print("Number of cases:", df.shape[0])
  print("Number of NaNs in each column:\n",gen_cols_with_nan(df))
  print("\n")


def prepare_data(data_fp, dtime_fp, exclude21=True):
  """
  Prepares the patient data, combining date-time related info
  """
  # Load dashboard dataset
  dashb_df = pd.read_csv(data_fp)
  print_df_info(dashb_df, dfname='Dashboard')
  # Drop rows with NaN in weight z-score
  dashb_df = dashb_df[globals.DASHDATA_COLS].dropna(subset=['WEIGHT_ZSCORE'])
  print_df_info(dashb_df, dfname='Processed Dashboard')


  # Load datetime dataset
  dtime_df = pd.read_csv(dtime_fp, parse_dates=['ADMIT_DATE', 'DISCHARGE_DATE', 'SURGEON_START_DT_TM',
       'SURGERY_END_DT_TM'])
  print_df_info(dtime_df, dfname='Datetime')
  # Drop rows that are NaN in selected columns
  dtime_df = dtime_df[globals.DATETIME_COLS].dropna(subset=globals.DATETIME_COLS)
  print_df_info(dtime_df, dfname='Processed Datetime')

  # Compute the number of nights
  admit_date, discharge_date = dtime_df['ADMIT_DATE'].dt.date, dtime_df['DISCHARGE_DATE'].dt.date
  dtime_df['NUM_OF_NIGHTS'] = (discharge_date - admit_date) / np.timedelta64(1, 'D')
  #print(dtime_df.head(20))

  # Combine with dashboard_df
  dashb_df = dashb_df.join(dtime_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='inner')
  print_df_info(dashb_df, dfname="Combined dashboard df")
  # print(dashb_df[dashb_df['NUM_OF_NIGHTS'].isnull()])
  # TODO: Very weird that case 62211921 is null at admit and discharge time, investigate this with csv reader

  # Load CPT and combine with existing hierarchy


  # Exclude 2021 data by request
  if exclude21:
    dashb_df = dashb_df[dashb_df['DISCHARGE_DATE'].dt.year < 2021]
    print_df_info(dashb_df, "Final Dashboard (2021 excluded)")

  return dashb_df


def gen_sps_data(df):
  """Select a subset of cases where SPS surgeon estimation is available"""
  # Q: When surgeons give an estimation, do they actually mean #nights?
  # A: According to Lynne, doctors mean number of nights.
  return df.loc[df['SPS_PREDICTED_LOS'].notnull()]


def gen_data_without_sps_pred(df):
  """Select a subset of cases where SPS surgeon estimation is not available"""
  return df.loc[df['SPS_PREDICTED_LOS'].isnull()]


def gen_cpt_group_one_hot(df, cpt2group_df):

  return


def eval_surgeon_perf(df, idxs=None, Xtype='all'):
  if idxs is not None:
    df = df.iloc[idxs]
    print(df.shape[0])

  # Evaluate SPS surgeon estimation performance
  true_nnt = dpp.gen_y_nnt(df['NUM_OF_NIGHTS'])
  surgeon_pred = dpp.gen_y_nnt(df['SPS_PREDICTED_LOS'])
  model_eval.gen_confusion_matrix(true_nnt, surgeon_pred, globals.SURGEON, isTrain=True if Xtype == 'training' else False)

  print(globals.SURGEON)
  print("Accuracy (%s): " % Xtype, metrics.accuracy_score(true_nnt, surgeon_pred, normalize=True))
  class_names = [str(i) for i in range(globals.MAX_NNT+1)] + ["%d+" % globals.MAX_NNT]
  f1_sps = pd.DataFrame({"NNT class": class_names, "F-1 score": metrics.f1_score(true_nnt, surgeon_pred, average=None)})
  rmse = np.sqrt(metrics.mean_squared_error(true_nnt, surgeon_pred))
  print("F-1 score (%s): " % Xtype, f1_sps)
  print("RMSE (%s)" % Xtype, rmse)

  figs, axs = plt.subplots(nrows=2, ncols=1, figsize=(18, 20))
  model_eval.gen_error_histogram(true_nnt, surgeon_pred, globals.SURGEON, Xtype=Xtype, yType="Number of nights", ax=axs[0],
                                 groupby_outcome=True)
  model_eval.gen_error_hist_pct(true_nnt, surgeon_pred, globals.SURGEON, Xtype=Xtype, yType="Number of nights", ax=axs[1])

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


# if __name__ == "__main__":
#   data_df = prepare_data(Path("../Data_new_all", "dashboard_data_18to21.csv"),
#                          Path("../Data_new_all", "dtime_data_18to21.csv"))
#
#   gen_sps_data(data_df)