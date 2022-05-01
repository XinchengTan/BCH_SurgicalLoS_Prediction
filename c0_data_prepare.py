"""
This script contains helper functions to prepare the raw dataset pulled from DB.
"""
import pandas as pd
import numpy as np
from pathlib import PosixPath

from globals import *


pd.set_option('display.max_columns', 50)


def prepare_data(data_fp, cpt_fp, cpt_grp_fp, ccsr_fp, medication_fp, dtime_fp=None, chews_fp=None,
                 care_class_fp=None, exclude2021=False, force_weight=False):
  """
  Prepares the patient LoS dataset, combining datetime and CPT related info
  """
  # Load dashboard dataset
  date_cols = [ADMIT_DTM, DISCHARGE_DTM, SURG_START_DTM, SURG_END_DTM]
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
    dashb_df = dashb_df.dropna(subset=['WEIGHT_ZSCORE'])
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
    dtime_df = dtime_df[DATETIME_COLS].dropna(subset=DATETIME_COLS)
    print_df_info(dtime_df, dfname='Processed Datetime DF')
    dashb_df = dashb_df.join(dtime_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='inner')
    print_df_info(dashb_df, dfname="Combined dashboard df")

  # Compute the number of nights
  if ADMIT_DTM in dashb_df.columns and DISCHARGE_DTM in dashb_df.columns:
    admit_date, discharge_date = dashb_df[ADMIT_DTM].dt.date, dashb_df[DISCHARGE_DTM].dt.date
    dashb_df[NNT] = (discharge_date - admit_date) / np.timedelta64(1, 'D')
  else:
    dashb_df[NNT] = 1000  # Placeholder outcome when it's not available

  # Handle basic NaNs
  dashb_df.fillna({os: 0.0 for os in OS_CODE_LIST}, inplace=True)
  #dashb_df.fillna({chews_3plus: False for chews_3plus in PHYSIO_DECLINE_SET})
  print_df_info(dashb_df, "Dashboard DF (filled NaN OS with 0, CHEWS decline with False)")
  dashb_df = dashb_df[dashb_df[NNT].notna()]
  dashb_df = dashb_df[dashb_df[STATE].notna()]
  print_df_info(dashb_df, "Dashboard DF (dropped rows with NaN in outcome & state code)")

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
    'CPT_CODE': lambda x: list(set(x)),
    'CPT_GROUP': lambda x: list(set(x))
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
    .agg({'ccsr_1': lambda x: list(set(x))})\
    .reset_index()\
    .rename(columns={'ccsr_1': 'CCSRS'})  # 'icd10': 'ICD10S', 'icd10': lambda x: list(x)

  # Join with dashboard df
  dashb_df = dashb_df.join(diags_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='left')
  dashb_df['CCSRS'] = dashb_df['CCSRS'].apply(lambda x: x if isinstance(x, list) else [])
  # dashb_df['ICD10S'] = dashb_df['ICD10S'].apply(lambda x: x if isinstance(x, list) else [])
  print_df_info(dashb_df, 'Dashboard DF with chronic conditions')

  # Join dashboard df with CHEWS df
  if chews_fp is not None:
    chews_df = pd.read_csv(chews_fp)
    dashb_df = add_chews_decline_outcomes(dashb_df=dashb_df, chews_df=chews_df, fillna_val=False)

  # Join dashboard df with care_class column
  if (CARE_CLASS not in dashb_df.columns) and (care_class_fp is not None):
    care_class_df = pd.read_csv(care_class_fp)
    dashb_df = dashb_df.join(care_class_df[['SURG_CASE_KEY', CARE_CLASS]].set_index('SURG_CASE_KEY'),
                             on='SURG_CASE_KEY',
                             how='inner')
    print_df_info(dashb_df, 'Dashboard DF with care class')

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


def active_med_df(med_df, keep_status=None):
  # 3: still taking, as listed; 4: still taking, not as prescribed; 10: confirmed, no home meds
  if keep_status is None:
    return med_df
  med_df = med_df.loc[med_df[DRUG_STATUS_KEY].isin(list(keep_status))]
  return med_df


def join_med_df_list_rep(med_df, dashb_df, levels=(1,), keep_status=(3, 4, 10)):
  # filter out inactive meds
  med_df = active_med_df(med_df, keep_status=keep_status)

  # get drug level for each medication
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


# Process CHEWS df to 1. obtain CHEWS decline flags for respiratory, cardiovascular and neurologic
def add_chews_decline_outcomes(dashb_df: pd.DataFrame, chews_df: pd.DataFrame, fillna_val=None):
  assert CHEWS_TYPE in chews_df.columns, f'{CHEWS_TYPE} is required in CHEWS_df!'
  # groupby surg_case_key and chews_type, get max chews score
  case_chews_df = chews_df.groupby(by=['SURG_CASE_KEY', CHEWS_TYPE])\
    .agg({'SCORE': 'max'})\
    .reset_index(level=[CHEWS_TYPE])
  # convert to bool columns with criterion: score > 2
  for decline_type in PHYSIO_DECLINE_SET:
    dashb_df = add_chews_decline_outcome(decline_type, dashb_df, case_chews_df, fillna_val=fillna_val)
    print_df_info(dashb_df, f'Dashboard DF with CHEWS outcome {decline_type}')
  return dashb_df


# Add a single CHEWS decline outcome boolean column to dashb_df
def add_chews_decline_outcome(chews_decline_type: str, dashb_df: pd.DataFrame, case_chews_agg_df: pd.DataFrame, fillna_val=None):
  assert chews_decline_type in PHYSIO_DECLINE_SET, f'CHEWS decline type must be one of {PHYSIO_DECLINE_SET}!'
  assert case_chews_agg_df.index.name == 'SURG_CASE_KEY', 'Index column of case_chews_agg_df must be "SURG_CASE_KEY"!'
  if chews_decline_type in dashb_df.columns:
    dashb_df.drop(columns=chews_decline_type, inplace=True)
  declineType2chewsLabel = {CARDIO_DECLINE: 'CHEWS-  Cardiovascular',
                            RESPIR_DECLINE: 'CHEWS-  Respiratory',
                            NEURO_DECLINE: 'CHEWS-  Behavior/Neuro'}

  case_chewsX_df = case_chews_agg_df[case_chews_agg_df[CHEWS_TYPE] == declineType2chewsLabel[chews_decline_type]]
  case_chewsX_df[chews_decline_type] = case_chewsX_df['SCORE'] > 2
  dashb_df = dashb_df.join(case_chewsX_df[[chews_decline_type]], on='SURG_CASE_KEY', how='left')
  if fillna_val is not None:
    dashb_df.fillna({chews_decline_type: fillna_val}, inplace=True)
  return dashb_df


def gen_sps_data(df):
  """Select a subset of cases where SPS surgeon estimation is available"""
  # Q: When surgeons give an estimation, do they actually mean #nights?
  # A: According to Lynne, doctors mean number of nights.
  return df.loc[df['SPS_PREDICTED_LOS'].notnull()]


def gen_data_without_sps_pred(df):
  """Select a subset of cases where SPS surgeon estimation is not available"""
  return df.loc[df['SPS_PREDICTED_LOS'].isnull()]
