"""
This script contains helper functions to prepare the raw dataset pulled from DB.
"""
from IPython.display import display
import pandas as pd
import numpy as np
from pathlib import Path, PosixPath

import globals
import c1_data_preprocessing as dpp
import utils_plot as pltutil


pd.set_option('display.max_columns', 50)


def prepare_data(data_fp, cpt_fp, cpt_grp_fp, ccsr_fp, medication_fp, dtime_fp=None,
                 exclude2021=False, force_weight=False):
  """
  Prepares the patient LoS dataset, combining datetime and CPT related info
  """
  # Load dashboard dataset
  date_cols = [globals.ADMIT_DTM, globals.DISCHARGE_DTM, globals.SURG_START_DTM, globals.SURG_END_DTM]
  if type(data_fp) == str or type(data_fp) == PosixPath:
    dashb_df = pd.read_csv(data_fp)
    available_date_cols = np.array(date_cols)[np.in1d(date_cols, dashb_df.columns)]
    if len(available_date_cols) > 0:
      dashb_df = pd.read_csv(data_fp, parse_dates=list(available_date_cols))
  elif isinstance(data_fp, pd.DataFrame):
    dashb_df = data_fp.copy()
  else:
    raise NotImplementedError("prepare data not implemented for this data_fp type")
  print_df_info(dashb_df, dfname='Dashboard')

  # Drop rows with NaN in weight z-score
  if force_weight:
    dashb_df = dashb_df[globals.DASHDATA_COLS].dropna(subset=['WEIGHT_ZSCORE'])
  print_df_info(dashb_df, dfname='Processed Dashboard (weight)')

  # Medication data -- join by surg case key
  if medication_fp is not None:
    med_df = pd.read_csv(medication_fp)
    dashb_df, _ = join_med_df_list_rep(med_df, dashb_df, levels=(1, 2, 3, 123))
    print_df_info(dashb_df, dfname='Dashboard df with Medication')

  # Load datetime dataset
  if dtime_fp is not None:
    dtime_df = pd.read_csv(dtime_fp, parse_dates=date_cols)
    print_df_info(dtime_df, dfname='Datetime DF')
    # Drop rows that are NaN in selected columns
    dtime_df = dtime_df[globals.DATETIME_COLS].dropna(subset=globals.DATETIME_COLS)
    print_df_info(dtime_df, dfname='Processed Datetime DF')
    dashb_df = dashb_df.join(dtime_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='inner')
    print_df_info(dashb_df, dfname="Combined dashboard df")

  # Compute the number of nights
  if globals.ADMIT_DTM in dashb_df.columns and globals.DISCHARGE_DTM in dashb_df.columns:
    admit_date, discharge_date = dashb_df[globals.ADMIT_DTM].dt.date, dashb_df[globals.DISCHARGE_DTM].dt.date
    dashb_df[globals.NNT] = (discharge_date - admit_date) / np.timedelta64(1, 'D')
  else:
    dashb_df[globals.NNT] = 1000  # Placeholder outcome when it's not available

  # Handle basic NaNs
  dashb_df.fillna({os: 0.0 for os in globals.OS_CODE_LIST}, inplace=True)
  print_df_info(dashb_df, "Dashboard DF (fill NaN OS with 0)")
  dashb_df = dashb_df[dashb_df[globals.NNT].notna()]
  dashb_df = dashb_df[dashb_df['STATE_CODE'].notna()]
  print_df_info(dashb_df, "Dashboard DF (dropped rows with NaN)")

  # Load CPTs for each case and combine with the existing hierarchy
  cpt_df, cpt_grp_df = pd.read_csv(cpt_fp), pd.read_csv(cpt_grp_fp)
  all_cases_cnt = cpt_df['SURG_CASE_KEY'].nunique()  # TODO: Dashboard df and CPT df does not match in surgical case keys
  print("All cases (CPT df): %d" % all_cases_cnt)

  # Join with CPT hierarchy group; discard cases if any of their CPT is not present in the existing hierarchy
  if cpt_df.dtypes['CPT_CODE'] == object:
    cpt_df = cpt_df[cpt_df['CPT_CODE'].apply(lambda x: x.isnumeric())]  # discard rows with non-numeric cpt codes
    cpt_df['CPT_CODE'] = cpt_df['CPT_CODE'].astype(int)
  cpt_df = cpt_df.join(cpt_grp_df.set_index('CPT_CODE'), on='CPT_CODE', how='inner')
  print_df_info(cpt_df, dfname='CPT with Group')
  # cpt_df = cpt_df.join(cpt_grp_df.set_index('CPT_CODE'), on='CPT_CODE', how='inner')
  # print_df_info(cpt_df, 'CPT with Group')

  cpt_df = cpt_df.groupby('SURG_CASE_KEY')\
    .agg({
    'CPT_CODE': lambda x: list(x),
    'CPT_GROUP': lambda x: list(x)
  })\
    .reset_index()
  print("\nDiscarded %d cases whose CPT(s) are all unknown!\n" % (all_cases_cnt - cpt_df.shape[0]))

  # Join case to list of CPTs with the dashboard data
  # okay to discard cases without CPTs, since it's required
  dashb_df = dashb_df.join(cpt_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='inner')\
    .rename(columns={'CPT_CODE': 'CPTS', 'CPT_GROUP': 'CPT_GROUPS'})
  print_df_info(dashb_df, dfname="Dashboard DF with CPT info")

  # Join with CCSR df
  diags_df, diag_codes = pd.read_csv(ccsr_fp), ['ccsr_1']  # ['ccsr_1', 'icd10']
  print_df_info(diags_df, "Diagnosis DF", other_cols=['ccsr_1'])
  diags_df = diags_df.dropna(axis=0, how='any', subset=['ccsr_1'])\
    .groupby('SURG_CASE_KEY')\
    .agg({'ccsr_1': lambda x: list(x)})\
    .reset_index()\
    .rename(columns={'ccsr_1': 'CCSRS'})  # 'icd10': 'ICD10S', 'icd10': lambda x: list(x)

  # Join with dashboard df
  dashb_df = dashb_df.join(diags_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='left')
  dashb_df['CCSRS'] = dashb_df['CCSRS'].apply(lambda x: x if isinstance(x, list) else [])
  # dashb_df['ICD10S'] = dashb_df['ICD10S'].apply(lambda x: x if isinstance(x, list) else [])
  print_df_info(dashb_df, 'Dashboard DF with chronic conditions')

  # Exclude 2021 data by request
  if exclude2021:
    dashb_df = dashb_df[dashb_df['DISCHARGE_DATE'].dt.year < 2021]
    print_df_info(dashb_df, "Final Dashboard (2021 excluded)")

  return dashb_df


def gen_cols_with_nan(df):
  res = df.isnull().sum().sort_values(ascending=False).head(20)
  return res


def print_df_info(df, dfname="Dashboard", other_cols=None):
  print("%s columns:\n" % dfname, df.keys())
  print("Number of cases: %d" % df['SURG_CASE_KEY'].nunique())
  if other_cols:
    for col_name in other_cols:
      print("Number of unique, non-NaN values in '%s': %d" % (col_name, df[col_name].nunique(dropna=True)))
  print("Number of NaNs in each column:\n%s" % gen_cols_with_nan(df))
  print("\n")


def join_med_df_list_rep(med_df, dashb_df, levels=(1,)):
  med_levels = [f'LEVEL{l}_DRUG_CLASS_NAME' for l in levels[:3]]
  med_df = med_df[['SURG_CASE_KEY', 'HNA_ORDER_MNEMONIC'] + med_levels]
  med_df.loc[med_df['LEVEL1_DRUG_CLASS_NAME'].isna(), med_levels] = med_df['HNA_ORDER_MNEMONIC']  # use mnemonic if level3 med == null
  if 123 in levels:  # composite level
    med_df['LEVEL123_DRUG_CLASS_NAME'] = med_df['LEVEL3_DRUG_CLASS_NAME']
    med_df.loc[med_df['LEVEL123_DRUG_CLASS_NAME'].isna(), 'LEVEL123_DRUG_CLASS_NAME'] = med_df['LEVEL2_DRUG_CLASS_NAME']
    med_df.loc[med_df['LEVEL123_DRUG_CLASS_NAME'].isna(), 'LEVEL123_DRUG_CLASS_NAME'] = med_df['LEVEL1_DRUG_CLASS_NAME']

  med_df = med_df.groupby(by='SURG_CASE_KEY')\
    .agg({f'LEVEL{l}_DRUG_CLASS_NAME': lambda x: list({d for d in x if not pd.isna(d)}) for l in levels})\
    .reset_index()
  dashb_df = dashb_df.join(med_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='left')
  for l in levels:
    dashb_df[f'LEVEL{l}_DRUG_CLASS_NAME'] = dashb_df[f'LEVEL{l}_DRUG_CLASS_NAME']\
      .apply(lambda x: x if isinstance(x, list) else [])
  # TODO: if levels starts at 2, and some are Nan at this level & are not DME, then what?
  return dashb_df, med_df


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