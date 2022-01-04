"""
This script contains helper functions to prepare the raw dataset pulled from DB.
"""
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from pathlib import Path
from sklearn import metrics

from . import globals, c4_model_eval
from . import c1_data_preprocessing as dpp
from . import utils_plot as pltutil


pd.set_option('display.max_columns', 50)


def gen_cols_with_nan(df):
  res = df.isnull().sum().sort_values(ascending=False)
  return res


def print_df_info(df, dfname="Dashboard", other_cols=None):
  print("%s columns:\n" % dfname, df.keys())
  print("Number of cases: %d" % df['SURG_CASE_KEY'].nunique())
  if other_cols:
    for col_name in other_cols:
      print("Number of unique, non-NaN values in '%s': %d" % (col_name, df[col_name].nunique(dropna=True)))
  print("Number of NaNs in each column:\n%s" % gen_cols_with_nan(df))
  print("\n")


def prepare_data(data_fp, dtime_fp, cpt_fp, cpt_grp_fp, diag_fp, exclude2021=True, gen_no_weight=False):
  """
  Prepares the patient LoS dataset, combining datetime and CPT related info
  """
  # Load dashboard dataset
  if type(data_fp) == str:
    dashb_df = pd.read_csv(data_fp)
  elif isinstance(data_fp, pd.DataFrame):
    dashb_df = data_fp.copy()
  else:
    raise NotImplementedError("prepare data not implemented for this data_fp type")
  print_df_info(dashb_df, dfname='Dashboard')

  # Drop rows with NaN in weight z-score
  if gen_no_weight:
    dashb_df_no_weight = dashb_df[dashb_df['WEIGHT_ZSCORE'].isna()][globals.DASHDATA_COLS]
    dashb_df_no_weight = prepare_data(dashb_df_no_weight, dtime_fp, cpt_fp, cpt_grp_fp, diag_fp, exclude2021, False)

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
  dtime_df[globals.NNT] = (discharge_date - admit_date) / np.timedelta64(1, 'D')

  # Combine with dashboard_df
  dashb_df = dashb_df.join(dtime_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='inner')
  print_df_info(dashb_df, dfname="Combined dashboard df")
  # print(dashb_df[dashb_df['NUM_OF_NIGHTS'].isnull()])
  # TODO: Very weird that case 62211921 is null at admit and discharge time, investigate this with csv reader

  # Load CPTs for each case and combine with the existing hierarchy
  cpt_df, cpt_grp_df = pd.read_csv(cpt_fp), pd.read_csv(cpt_grp_fp)
  all_cases_cnt = cpt_df['SURG_CASE_KEY'].nunique()  # TODO: Dashboard df and CPT df does not match in surgical case keys
  print("All cases (CPT df): %d" % all_cases_cnt)
  # Join with CPT hierarchy group; discard cases if any of their CPT is not present in the existing hierarchy
  cpt_df = cpt_df.join(cpt_grp_df.set_index('CPT_CODE'), on='CPT_CODE', how='inner')
  print_df_info(cpt_df, 'CPT with Group')

  cpt_df = cpt_df.groupby('SURG_CASE_KEY')\
    .agg({
    'CPT_CODE': lambda x: list(x),
    'length_of_stay_decile': lambda x: list(x),
    'CPT_GROUP': lambda x: list(x)
  })\
    .reset_index()
  print("\nDiscarded %d cases whose CPT(s) are all unknown!\n" % (all_cases_cnt - cpt_df.shape[0]))

  # Join case to list of CPTs with the dashboard data  TODO: What to do with cases without any CPTs? Is it normal?
  dashb_df = dashb_df.join(cpt_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='inner')\
    .rename(columns={'CPT_CODE': 'CPTS',
             'length_of_stay_decile': 'CPT_DECILES',
             'CPT_GROUP': 'CPT_GROUPS'})
  print_df_info(dashb_df, dfname="Dashboard DF with CPT info")
  # left join: 18654; inner join: 17269

  # Join with CCSR df
  diags_df = pd.read_csv(diag_fp)
  print_df_info(diags_df, "Diagnosis DF", other_cols=['ccsr_1', 'icd10'])
  diags_df = diags_df.dropna(axis=0, how='any', subset=['ccsr_1', 'icd10'])\
    .groupby('SURG_CASE_KEY')\
    .agg({'ccsr_1': lambda x: list(x),
          'icd10': lambda x: list(x)})\
    .reset_index()\
    .rename(columns={'ccsr_1': 'CCSRS', 'icd10': 'ICD10S'})

  # Join with dashboard df
  # TODO: [BUG] Should be left join & replace NAN with [], since we still need the cases without any ICD10s?
  dashb_df = dashb_df.join(diags_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='left')
  dashb_df['CCSRS'] = dashb_df['CCSRS'].apply(lambda x: x if isinstance(x, list) else [])
  dashb_df['ICD10S'] = dashb_df['ICD10S'].apply(lambda x: x if isinstance(x, list) else [])
  print_df_info(dashb_df, 'Dashboard DF with chronic conditions')
  # Inner join:
  # Number of cases: 15748
  # Number of NaNs in each column:
  # SPS_PREDICTED_LOS      7697

  # Left join:  # TODO: replace nan with empty list (pd.fillna() does not accept list??????)
  # Number of cases: 17262
  # Number of NaNs in each column:
  # SPS_PREDICTED_LOS      8668
  # ICD10S                 1514
  # CCSRS                  1514

  # Exclude 2021 data by request
  if exclude2021:
    dashb_df = dashb_df[dashb_df['DISCHARGE_DATE'].dt.year < 2021]
    print_df_info(dashb_df, "Final Dashboard (2021 excluded)")

  return dashb_df if not gen_no_weight else dashb_df, dashb_df_no_weight


def gen_data_without_weight(df):

  return


def gen_sps_data(df):
  """Select a subset of cases where SPS surgeon estimation is available"""
  # Q: When surgeons give an estimation, do they actually mean #nights?
  # A: According to Lynne, doctors mean number of nights.
  return df.loc[df['SPS_PREDICTED_LOS'].notnull()]


def gen_data_without_sps_pred(df):
  """Select a subset of cases where SPS surgeon estimation is not available"""
  return df.loc[df['SPS_PREDICTED_LOS'].isnull()]



# if __name__ == "__main__":
#   data_df = prepare_data(Path("../Data_new_all", "dashboard_data_18to21.csv"),
#                          Path("../Data_new_all", "dtime_data_18to21.csv"))
#
#   gen_sps_data(data_df)