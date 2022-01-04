"""
This script provides a full pipeline that
  1. ingests the surgical patient dataset from a csv file,
  2. preprocesses the dataset into numerical matrices,
  3. applies pre-trained machine learning models to predict LoS outcome.

Author: Cara (Xincheng) Tan (xinchengtan@g.harvard.edu)
"""

import numpy as np
import pandas as pd
import pickle


PRETRAINED_MODELS_FP = './pretrained_models'
O2M_CASES_FP = './historical_info/o2m_cases.csv'

NNT = "NUM_OF_NIGHTS"
SPS_LOS_FTR = 'SPS_PREDICTED_LOS'
CCSRS = 'CCSRS'
ICD10S = 'ICD10S'
CPTS = 'CPTS'

DASHDATA_COLS_TRAINING = ['SURG_CASE_KEY', 'LENGTH_OF_STAY', 'SPS_PREDICTED_LOS', 'BOOKING_DATE',
                          'SEX_CODE', 'AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE', 'PROC_DECILE',
                          'Endocrine', 'Genetic', 'Hematologic', 'Immunologic', 'Infectious', 'Mental',
                          'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic', 'Nutrition', 'Optic',
                          'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin', 'Uncategorized', 'Urogenital']

DASHDATA_COLS_PREDICT = ['SURG_CASE_KEY', 'LENGTH_OF_STAY', 'SPS_PREDICTED_LOS', 'BOOKING_DATE',
                         'SEX_CODE', 'AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE', 'PROC_DECILE',
                         'Endocrine', 'Genetic', 'Hematologic', 'Immunologic', 'Infectious', 'Mental',
                         'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic', 'Nutrition', 'Optic',
                         'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin', 'Uncategorized', 'Urogenital']

DATETIME_COLS_TRAINING = ['SURG_CASE_KEY', 'CARE_CLASS', 'ICU_BED_NEEDED', 'PRIMARY_PROC',
                          'ADMIT_DATE', 'DISCHARGE_DATE', 'SURGEON_START_DT_TM', 'SURGERY_END_DT_TM']

DATETIME_COLS_PREDICT = ['SURG_CASE_KEY', 'PRIMARY_PROC']

FEATURE_COLS_NO_WEIGHT = ['SURG_CASE_KEY', 'SEX_CODE', 'AGE_AT_PROC_YRS', 'PROC_DECILE',
                          'Endocrine', 'Genetic', 'Hematologic', 'Immunologic', 'Infectious', 'Mental',
                          'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic', 'Nutrition', 'Optic',
                          'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin', 'Uncategorized', 'Urogenital',
                          CPTS, 'CPT_GROUPS', 'PRIMARY_PROC', CCSRS, ICD10S]

NON_NUMERIC_COLS = ['SURG_CASE_KEY', CPTS, 'CPT_GROUPS', 'PRIMARY_PROC', CCSRS, ICD10S]

AGE_BINS = [0, 0.25, 0.5, 1, 2, 3, 6, 9, 12, 15, 18, float('inf')]
WEIGHT_Z_BINS = [float('-inf'), -4, -2, -1, 1, 2, 4, float('inf')]

MAX_NNT = 5

# TODO [IP for meeting] 1. Drop data without SPS prediction???
# TODO 2. Produce a unique dataset to be compared with -- think about the file size and storage format  .npy?
# TODO 3. Remove identical cases with different outcomes


# Data prepare (filter out cases with missing information)
def get_combined_dataframe(data_fp, dtime_fp, cpt_fp, cpt_grp_fp, diag_fp, for_training=False, verbose=False):
  """
  Prepares the patient LoS dataset, combining datetime and CPT related info
  """
  # Load dashboard dataset
  if type(data_fp) == str:
    dashb_df = pd.read_csv(data_fp)
  elif isinstance(data_fp, pd.DataFrame):
    dashb_df = data_fp.copy()
  else:
    raise NotImplementedError("get_combined_dataframe() does not support this 'data_fp' type")
  print_df_info(dashb_df, dfname='Dashboard', verbose=verbose)

  # Define dataframe columns to be extracted
  dash_cols, date_cols = (DASHDATA_COLS_PREDICT, DATETIME_COLS_PREDICT) if not for_training \
    else (DASHDATA_COLS_TRAINING, DATETIME_COLS_TRAINING)

  if 'WEIGHT_ZSCORE' in dash_cols:
    dashb_df = dashb_df[dash_cols].dropna(subset=['WEIGHT_ZSCORE'])
    print_df_info(dashb_df, dfname='Dashboard df with valid weight z-score', verbose=verbose)

  # Load datetime dataset
  dtime_df = pd.read_csv(dtime_fp, parse_dates=['ADMIT_DATE', 'DISCHARGE_DATE', 'SURGEON_START_DT_TM',
                                                'SURGERY_END_DT_TM']) if for_training else pd.read_csv(dtime_fp)
  print_df_info(dtime_df, dfname='Datetime', verbose=verbose)

  # Drop rows that are NaN in selected columns
  dtime_df = dtime_df[date_cols].dropna(subset=date_cols)
  print_df_info(dtime_df, dfname='Datetime df with NA removed', verbose=verbose)

  if for_training:
    # Compute the number of nights (response vector)
    admit_date, discharge_date = dtime_df['ADMIT_DATE'].dt.date, dtime_df['DISCHARGE_DATE'].dt.date
    dtime_df[NNT] = (discharge_date - admit_date) / np.timedelta64(1, 'D')

  # TODO: Should I drop rows without SPS prediction??


  # Combine with dashboard_df
  dashb_df = dashb_df.join(dtime_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='inner')
  print_df_info(dashb_df, dfname='Dashboard df + Datetime df', verbose=verbose)

  # Load CPTs for each case and combine with the existing hierarchy
  cpt_df, cpt_grp_df = pd.read_csv(cpt_fp), pd.read_csv(cpt_grp_fp)
  all_cases_cnt = cpt_df['SURG_CASE_KEY'].nunique()

  # Join with CPT hierarchy group; discard cases if none of their CPTs is present in the existing hierarchy
  cpt_df = cpt_df.join(cpt_grp_df.set_index('CPT_CODE'), on='CPT_CODE', how='inner')
  print_df_info(cpt_df, dfname='CPT with Group', verbose=verbose)

  cpt_df = cpt_df.groupby('SURG_CASE_KEY')\
    .agg({
    'CPT_CODE': lambda x: list(x),
    'length_of_stay_decile': lambda x: list(x),
    'CPT_GROUP': lambda x: list(x)
  })\
    .reset_index()
  print("\nDiscarded %d cases whose CPT(s) are all unknown!\n" % (all_cases_cnt - cpt_df.shape[0]))

  # Join CPT df with the dashboard df by case key
  dashb_df = dashb_df.join(cpt_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='inner')\
    .rename(columns={'CPT_CODE': 'CPTS',
             'length_of_stay_decile': 'CPT_DECILES',
             'CPT_GROUP': 'CPT_GROUPS'})
  print_df_info(dashb_df, dfname="Dashboard DF with CPT info", verbose=verbose)

  # Join with CCSR df
  diags_df = pd.read_csv(diag_fp)
  print_df_info(diags_df, "Diagnosis DF", other_cols=['ccsr_1', 'icd10'])
  diags_df = diags_df.dropna(axis=0, how='any', subset=['ccsr_1', 'icd10'])\
    .groupby('SURG_CASE_KEY')\
    .agg({'ccsr_1': lambda x: list(x),
          'icd10': lambda x: list(x)})\
    .reset_index()\
    .rename(columns={'ccsr_1': CCSRS, 'icd10': ICD10S})

  # Join with dashboard df, and use [] as default value for CCSRS and ICD10S
  dashb_df = dashb_df.join(diags_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='left')
  dashb_df[CCSRS] = dashb_df[CCSRS].apply(lambda x: x if isinstance(x, list) else [])
  dashb_df[ICD10S] = dashb_df[ICD10S].apply(lambda x: x if isinstance(x, list) else [])
  print_df_info(dashb_df, dfname='Dashboard DF with chronic conditions', verbose=verbose)

  return dashb_df


# helper function that logs the intermediate dataframe info at each filtering step in get_combined_dataframe()
def print_df_info(df, dfname="Dashboard", other_cols=None, verbose=False):
  if verbose:
    print("%s columns:\n" % dfname, df.keys())
    print("Number of cases: %d" % df['SURG_CASE_KEY'].nunique())
    if other_cols:
      for col_name in other_cols:
        print("Number of unique, non-NaN values in '%s': %d" % (col_name, df[col_name].nunique(dropna=True)))
    print("Number of NaNs in each column:\n%s" % df.isnull().sum().sort_values(ascending=False))
    print("\n")


# Data preprocessing (clean, discretize and one-hot encode certain features)
def get_feature_matrix(df, cols=FEATURE_COLS_NO_WEIGHT, onehot_cols=None, onehot_dtypes=None, trimmed_ccsr=None,
                       discretize_cols=None):
  # Make data matrix X numeric
  X = df.copy()[cols]
  X.loc[(X.SEX_CODE != 'F'), 'SEX_CODE'] = 0.0
  X.loc[(X.SEX_CODE == 'F'), 'SEX_CODE'] = 1.0

  feature_cols = list(cols)
  if trimmed_ccsr:
    # add a column with only the target set of CCSRs
    if CCSRS in onehot_cols:
      X['Trimmed_CCSRS'] = X[CCSRS].apply(lambda row: [cc for cc in row if cc in trimmed_ccsr])
      onehot_cols = list(map(lambda item: item.replace(CCSRS, 'Trimmed_CCSRS'), onehot_cols))
      feature_cols = list(map(lambda item: item.replace(CCSRS, 'Trimmed_CCSRS'), feature_cols))
    # add a column with only the ICD10s of the target set of CCSRs
    if ICD10S in onehot_cols:
      X['Trimmed_ICD10S'] = X[[CCSRS, ICD10S]].apply(lambda row: [row[ICD10S][i]
                                                                  for i in range(len(row[ICD10S]))
                                                                  if row[CCSRS][i] in trimmed_ccsr], axis=1)
      onehot_cols = list(map(lambda item: item.replace(ICD10S, 'Trimmed_ICD10S'), onehot_cols))
      feature_cols = list(map(lambda item: item.replace(ICD10S, 'Trimmed_ICD10S'), feature_cols))

  if onehot_cols is not None:
    # Apply one-hot encoding to the designated columns
    for oh_col, dtype in zip(onehot_cols, onehot_dtypes):
      if dtype == str:  # can directly use get_dummies()
        dummies = pd.get_dummies(X[oh_col], prefix=oh_col)
      elif dtype == list:  # Need to expand list to (row_id, oh_col indicator) first
        s = X[oh_col].explode()
        dummies = pd.crosstab(s.index, s).add_prefix(oh_col[:-1] + '_')
        dummies[dummies > 1] = 1  # in case a list contains duplicates  TODO: double check
      else:
        raise NotImplementedError("Please specify either a single column name (str) or a list of column names (list)")
      X = X.drop(columns=[oh_col]).join(dummies).fillna(0)
      feature_cols.remove(oh_col)
      feature_cols.extend(dummies.columns.to_list())

  # Save SURG_CASE_KEY, but drop with other non-numeric columns for data matrix
  X_case_key = X['SURG_CASE_KEY'].to_numpy()
  X.drop(columns=NON_NUMERIC_COLS, inplace=True, errors='ignore')
  for nnm_col in NON_NUMERIC_COLS:
    if nnm_col in feature_cols:
      feature_cols.remove(nnm_col)

  # Bucket SPS predicted LoS into 9 classes, if there such prediction exists
  if SPS_LOS_FTR in cols:  # Assume SPS prediction are all integers?
    X.loc[(X[SPS_LOS_FTR] > MAX_NNT), SPS_LOS_FTR] = MAX_NNT + 1

  # Discretize certain continuous columns by request
  X = X.to_numpy(dtype=np.float64)
  if discretize_cols:
    discretize_columns(X, feature_cols, discretize_cols, inplace=True)

  #

  # Basic sanity check
  assert X.shape[1] == len(feature_cols), 'Generated data matrix has %d features, but feature list has %d items' % \
                                          (X.shape[1], len(feature_cols))
  assert len(set(feature_cols)) == len(feature_cols), "Generated data matrix contains duplicated feature names!"

  return X, feature_cols, X_case_key


# helper function that removes cases that historically has multiple possible outcomes
def dump_o2m_cases(df):

  return

# helper function that discretizes age or weight z-score features
def discretize_columns(X, feature_names, discretize_cols, inplace=False):
  if not inplace:
    X = np.copy(X)
  # Modify data matrix with discretized columns by request
  for dis_col in discretize_cols:
    idx = feature_names.index(dis_col)
    if dis_col == 'AGE_AT_PROC_YRS':
      X[:, idx] = np.digitize(X[:, idx], AGE_BINS)
    elif dis_col == 'WEIGHT_ZSCORE':
      X[:, idx] = np.digitize(X[:, idx], WEIGHT_Z_BINS)
    else:
      raise Warning("%s discretization is not available yet!" % dis_col)
  return X


# Load pre-trained model
def load_pretrained_model(model_dir, model_fname):
  model_fp = model_dir + '/' + model_fname
  model = pickle.load(open(model_fp, 'rb'))
  return model


# Use pretrained model to predict outcome
def predict_los(X, sk_model):
  prediction = sk_model.predict(X)
  return prediction



if __name__ == "__main__":

  pass



