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

LOS = 'LENGTH_OF_STAY'
NNT = 'NUM_OF_NIGHTS'
SPS_LOS_FTR = 'SPS_PREDICTED_LOS'
CCSRS = 'CCSRS'
ICD10S = 'ICD10S'
CPTS = 'CPTS'

DASHDATA_COLS_TRAINING = ['SURG_CASE_KEY', 'LENGTH_OF_STAY', 'SPS_PREDICTED_LOS', 'PRIMARY_PROC',
                          'SEX_CODE', 'AGE_AT_PROC_YRS', 'PROC_DECILE',
                          'Endocrine', 'Genetic', 'Hematologic', 'Immunologic', 'Infectious', 'Mental',
                          'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic', 'Nutrition', 'Optic',
                          'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin', 'Uncategorized', 'Urogenital',
                          'HAR_ADMIT_DATE', 'HAR_DISCHARGE_DATE', 'CSN']  # 'WEIGHT_ZSCORE',

DASHDATA_COLS_PREDICT = ['SURG_CASE_KEY', 'LENGTH_OF_STAY', 'SPS_PREDICTED_LOS', 'PRIMARY_PROC',
                         'SEX_CODE', 'AGE_AT_PROC_YRS', 'PROC_DECILE',
                         'Endocrine', 'Genetic', 'Hematologic', 'Immunologic', 'Infectious', 'Mental',
                         'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic', 'Nutrition', 'Optic',
                         'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin', 'Uncategorized', 'Urogenital',
                         'HAR_ADMIT_DATE', 'HAR_DISCHARGE_DATE', 'CSN']  # 'WEIGHT_ZSCORE', 'BOOKING_DATE'

DATETIME_COLS_TRAINING = ['SURG_CASE_KEY', 'CARE_CLASS', 'ICU_BED_NEEDED', 'PRIMARY_PROC',
                          'ADMIT_DATE', 'DISCHARGE_DATE', 'SURGEON_START_DT_TM', 'SURGERY_END_DT_TM']

DATETIME_COLS_PREDICT = ['SURG_CASE_KEY', 'PRIMARY_PROC']

FEATURE_COLS_NO_WEIGHT = ['SURG_CASE_KEY', 'SEX_CODE', 'AGE_AT_PROC_YRS', 'PROC_DECILE',
                          'Endocrine', 'Genetic', 'Hematologic', 'Immunologic', 'Infectious', 'Mental',
                          'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic', 'Nutrition', 'Optic',
                          'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin', 'Uncategorized', 'Urogenital',
                          CPTS, 'CPT_GROUPS', 'PRIMARY_PROC', CCSRS]

NON_NUMERIC_COLS = ['SURG_CASE_KEY', CPTS, 'CPT_GROUPS', 'PRIMARY_PROC', CCSRS]

AGE_BINS = [0, 0.25, 0.5, 1, 2, 3, 6, 9, 12, 15, 18, float('inf')]
WEIGHT_Z_BINS = [float('-inf'), -4, -2, -1, 1, 2, 4, float('inf')]

MAX_NNT = 5

# TODO [IP for meeting] 1. Drop data without SPS prediction???
# TODO 2. Produce a unique dataset to be compared with -- think about the file size and storage format  .npy?
# TODO 3. Remove identical cases with different outcomes


# Data prepare (filter out cases with missing information)
def get_combined_dataframe(data_fp, cpt_fp, cpt_grp_fp, diag_fp, for_training=False, verbose=False):
  """
  Prepares the patient LoS dataset, combining datetime and CPT related info
  """
  # Load dashboard dataset
  dashb_df = pd.read_csv(data_fp, parse_dates=['HAR_ADMIT_DATE', 'HAR_DISCHARGE_DATE'])
  print_df_info(dashb_df, dfname='Dashboard', verbose=verbose)

  # Define the required columns
  dash_cols, date_cols = (DASHDATA_COLS_PREDICT, DATETIME_COLS_PREDICT) if not for_training \
    else (DASHDATA_COLS_TRAINING, DATETIME_COLS_TRAINING)

  # Drop rows with invalid weight z-score, if weight z-score is a required feature in the models
  if 'WEIGHT_ZSCORE' in dash_cols:
    dashb_df = dashb_df[dash_cols].dropna(subset=['WEIGHT_ZSCORE'])
    print_df_info(dashb_df, dfname='Dashboard df with valid weight z-score', verbose=verbose)

  # Bucket SPS predicted LoS into (MAX_NNT + 2) classes, if there such prediction exists
  if SPS_LOS_FTR in dash_cols:  # Assume SPS prediction are all integers?
    dashb_df.loc[(dashb_df[SPS_LOS_FTR] > MAX_NNT), SPS_LOS_FTR] = MAX_NNT + 1.0

  # Compute number of nights
  # set a cutoff at midnight i.e. discharges after midnight count toward an extra night of stay
  if for_training:
    dashb_df[NNT] = (dashb_df['HAR_DISCHARGE_DATE'].dt.date - dashb_df['HAR_ADMIT_DATE'].dt.date) / np.timedelta64(1, 'D')
    dashb_df.loc[(dashb_df[NNT] > MAX_NNT), NNT] = MAX_NNT + 1.0
    print_df_info(dashb_df, dfname='Dashboard df with NNT calculated', verbose=verbose)

  # Load CPTs for each case and combine with the existing hierarchy
  cpt_df, cpt_grp_df = pd.read_csv(cpt_fp), pd.read_csv(cpt_grp_fp)
  all_cases_cnt = cpt_df['SURG_CASE_KEY'].nunique()

  # Join with CPT hierarchy group; discard cases if none of their CPTs is present in the existing hierarchy
  cpt_df = cpt_df.join(cpt_grp_df.set_index('CPT_CODE'), on='CPT_CODE', how='inner')
  print_df_info(cpt_df, dfname='CPT with Group', verbose=verbose)

  cpt_df = cpt_df.groupby('SURG_CASE_KEY')\
    .agg({
    'CPT_CODE': lambda x: list(x),
    'CPT_GROUP': lambda x: list(x)
  })\
    .reset_index()
  print("\nDiscarded %d cases whose CPT(s) are all unknown!\n" % (all_cases_cnt - cpt_df.shape[0]))

  # Join CPT df with the dashboard df by case key
  dashb_df = dashb_df.join(cpt_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='left')\
    .rename(columns={'CPT_CODE': 'CPTS',
             'CPT_GROUP': 'CPT_GROUPS'})
  dashb_df['CPTS'] = dashb_df['CPTS'].apply(lambda x: x if isinstance(x, list) else [])
  dashb_df['CPT_GROUPS'] = dashb_df['CPT_GROUPS'].apply(lambda x: x if isinstance(x, list) else [])
  print_df_info(dashb_df, dfname="Dashboard DF with CPT info", verbose=verbose)

  # Join with CCSR df
  diags_df = pd.read_csv(diag_fp)
  print_df_info(diags_df, "Diagnosis DF", other_cols=['ccsr_1'])
  diags_df = diags_df.dropna(axis=0, how='any', subset=['ccsr_1'])\
    .groupby('SURG_CASE_KEY')\
    .agg({'ccsr_1': lambda x: list(x)})\
    .reset_index()\
    .rename(columns={'ccsr_1': CCSRS})

  # Join with dashboard df, and use [] as default value for CCSRS and ICD10S
  dashb_df = dashb_df.join(diags_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='left')
  dashb_df[CCSRS] = dashb_df[CCSRS].apply(lambda x: x if isinstance(x, list) else [])
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
def get_feature_matrix(df, skip_cases_fp, cols=FEATURE_COLS_NO_WEIGHT, onehot_cols=None, onehot_dtypes=None,
                       trimmed_ccsr=None, discretize_cols=None):
  # Get target feature names from training data
  target_features = pd.read_csv(skip_cases_fp).columns.to_list()

  # Make data matrix X numeric
  X = df.copy()[cols]
  X.loc[(X.SEX_CODE != 'F'), 'SEX_CODE'] = 0.0
  X.loc[(X.SEX_CODE == 'F'), 'SEX_CODE'] = 1.0

  if trimmed_ccsr:
    # add a column with only the target set of CCSRs
    if CCSRS in onehot_cols:
      X['Trimmed_CCSRS'] = X[CCSRS].apply(lambda row: [cc for cc in row if cc in trimmed_ccsr])
      onehot_cols = list(map(lambda item: item.replace(CCSRS, 'Trimmed_CCSRS'), onehot_cols))
    # add a column with only the ICD10s of the target set of CCSRs
    if ICD10S in onehot_cols:
      X['Trimmed_ICD10S'] = X[[CCSRS, ICD10S]].apply(lambda row: [row[ICD10S][i]
                                                                  for i in range(len(row[ICD10S]))
                                                                  if row[CCSRS][i] in trimmed_ccsr], axis=1)
      onehot_cols = list(map(lambda item: item.replace(ICD10S, 'Trimmed_ICD10S'), onehot_cols))

  print("Checkpoint 1 X shape: ", X.shape)

  # One-hot encode certain columns according to a given historical set of features (e.g. CPT, CCSR etc.)
  if onehot_cols is not None:
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
    # Drop rows that has certain indicator columns not covered in the target feature list (e.g. an unseen CPT code)
    new_ftrs = set(X.columns.to_list()) - set(target_features) - {'SURG_CASE_KEY', 'CPT_GROUPS'}
    if len(new_ftrs) > 0:
      case_idxs_with_new_features = set()
      for new_ftr in new_ftrs:
        idxs = X.index[X[new_ftr] == 1].to_list()
        case_idxs_with_new_features = case_idxs_with_new_features.union(set(idxs))
      X = X.drop(index=list(case_idxs_with_new_features))\
        .drop(columns=list(new_ftrs))\
        .reset_index(drop=True)
      print("Dropped %d cases with new features" % len(list(case_idxs_with_new_features)))
      print("Unseen codes: ", new_ftrs)
      if X.shape[0] == 0:
        raise Exception("All cases in this dataset contain an unseen indicator feature!")
    print("\nTarget columns#: ", len(target_features))
    print("Target columns #CCSRs: ", len(list(filter(lambda x: x.startswith('CCSR_'), target_features))))
    print("Target columns #CPTs: ", len(list(filter(lambda x: x.startswith('CPT_'), target_features))))
    print("Target columns #PProcs: ", len(list(filter(lambda x: x.startswith('PRIMARY_PROC_'), target_features))))

    print("\nCheckpoint 2 X shape: ", X.shape)
    ccsr_cols = list(filter(lambda x: x.startswith('CCSR_'), X.columns.to_list()))
    print(X[ccsr_cols].sum(axis=0))
    print("#CCSR indicators: ", len(ccsr_cols))
    print("#CPT indicators: ", len(list(filter(lambda x: x.startswith('CPT_'), X.columns.to_list()))))
    print("#PPROC indicators: ", len(list(filter(lambda x: x.startswith('PRIMARY_PROC_'), X.columns.to_list()))))
    # Add uncovered indicators as columns
    uncovered_ftrs = set(target_features) - set(X.columns.to_list())
    print("\nTotal un-occurred features: ", len(uncovered_ftrs))
    print("Total un-occurred CCSRs: ", len(list(filter(lambda x: x.startswith('CCSR_'), uncovered_ftrs))))
    print("Total un-occurred CPTs: ", len(list(filter(lambda x: x.startswith('CPT_'), uncovered_ftrs))))
    print("Total un-occurred PProcs: ", len(list(filter(lambda x: x.startswith('PRIMARY_PROC_'), uncovered_ftrs))))
    X[list(uncovered_ftrs)] = 0.0
    print('\nCCSR occurrence after adding un-occurred code indicators: ', X[ccsr_cols].sum(axis=0))

  print("\nCheckpoint 3 X shape: ", X.shape)
  # Save SURG_CASE_KEY, but drop with other non-numeric columns for data matrix
  X_case_key = X['SURG_CASE_KEY'].to_numpy()
  X.drop(columns=NON_NUMERIC_COLS, inplace=True, errors='ignore')
  print("Checkpoint 4 X shape: ", X.shape)

  # Reorder columns according to target_features
  X = X[target_features]
  print("Checkpoint 5 X shape: ", X.shape)

  # Discretize certain continuous columns by request
  X = X.to_numpy(dtype=np.float64)
  if discretize_cols:
    discretize_columns(X, target_features, discretize_cols, inplace=True)
  print("Checkpoint 6 X shape: ", X.shape)

  # Remove cases identical to any in skip_cases
  X, X_case_key = discard_o2m_cases_from_historical_data(X, X_case_key, skip_cases_fp)  # TODO: update X_case_keys with this
  print("Checkpoint 7 X shape: ", X.shape)

  # Basic sanity check
  assert X.shape[1] == len(target_features), 'Generated data matrix has %d features, but feature list has %d items' % \
                                          (X.shape[1], len(target_features))
  assert len(set(target_features)) == len(target_features), "Generated data matrix contains duplicated feature names!"

  return X, target_features, X_case_key


def get_outcome_vec(df, X_case_keys):
  return df[df['SURG_CASE_KEY'].isin(X_case_keys)][NNT].to_numpy()


def get_surgeon_pred_vec(df, X_case_keys):
  return df[df['SURG_CASE_KEY'].isin(X_case_keys)][SPS_LOS_FTR].to_numpy()


# Removes cases that historically has multiple possible outcomes
def discard_o2m_cases_from_historical_data(X, X_case_keys, skip_cases_fp):
  # skip_cases_fp: file path of the historical cases that have multiple possible outcomes
  skip_cases_df = pd.read_csv(skip_cases_fp, index_col=False)
  assert skip_cases_df.shape[1] == X.shape[1], "Input data matrix must match with training data in features!"

  Xdf = pd.DataFrame(X, columns=skip_cases_df.columns)
  Xdf['SURG_CASE_KEY'] = X_case_keys  # Maintain the corresponding surgical case keys
  # Left join minus inner join with skip_cases_df
  Xdf = pd.merge(Xdf, skip_cases_df, how='left', on=skip_cases_df.columns.to_list(), indicator=True)
  Xdf = Xdf[Xdf['_merge'] == 'left_only']

  return Xdf[skip_cases_df.columns].to_numpy(), Xdf['SURG_CASE_KEY'].to_numpy()


# Discretizes age or weight z-score features
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



