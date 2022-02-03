"""
This script provides a full pipeline that
  1. ingests the surgical patient dataset from a csv file,
  2. preprocesses the dataset into numerical matrices,
  3. applies a pre-trained machine learning model to predict the LoS outcome.

Author: Cara (Xincheng) Tan (xinchengtan@g.harvard.edu)
"""
import joblib
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import Dict

from . import los_prediction_global_vars as globals
from . import los_prediction_ensemble_model as ens


# Data prepare (filter out cases with missing information)
def get_combined_dataframe(data_fp, cpt_fp, cpt_grp_fp, diag_fp, pproc_fp, for_deployment_eval=False, verbose=False):
  """
  Prepares the patient LoS dataset, combining CPT & CCSR related info
  """
  # Load dashboard dataset
  if for_deployment_eval:
    dashb_df = pd.read_csv(data_fp, parse_dates=['HAR_ADMIT_DATE', 'HAR_DISCHARGE_DATE'])
    dash_cols = globals.DASHDATA_COLS_DEPLOY_EVAL  # Define the required columns

    # Bucket SPS predicted LoS into (MAX_NNT + 2) classes, if there such prediction exists
    # assume SPS predictions are all integers
    if globals.SPS_LOS_FTR in dash_cols:
      dashb_df.loc[(dashb_df[globals.SPS_LOS_FTR] > globals.MAX_NNT), globals.SPS_LOS_FTR] = globals.MAX_NNT + 1.0

    # Compute number of nights
    # set a cutoff at midnight i.e. discharges after midnight count toward an extra night of stay
    dashb_df[globals.NNT] = (dashb_df['HAR_DISCHARGE_DATE'].dt.date - dashb_df['HAR_ADMIT_DATE'].dt.date) / np.timedelta64(1, 'D')
    dashb_df.loc[(dashb_df[globals.NNT] > globals.MAX_NNT), globals.NNT] = globals.MAX_NNT + 1.0
    print_df_info(dashb_df, dfname='Dashboard df with NNT calculated and formatted', verbose=verbose)

    # Drop rows whose outcome == NaN
    dashb_df = dashb_df.dropna(subset=[globals.NNT])
    print_df_info(dashb_df, dfname='Dashboard DF (rows with NA outcome are dropped)', verbose=verbose)
  else:
    dashb_df = pd.read_csv(data_fp)
    dash_cols = globals.DASHDATA_COLS_PREDICT  # Define the required columns
  print_df_info(dashb_df, dfname='Dashboard df', verbose=verbose)

  # Drop rows with invalid weight z-score, if weight z-score is a required feature in the models
  if 'WEIGHT_ZSCORE' in dash_cols:
    dashb_df = dashb_df[dash_cols].dropna(subset=['WEIGHT_ZSCORE'])
    print_df_info(dashb_df, dfname='Dashboard df with valid weight z-score', verbose=verbose)

  # Add primary procedure decile column (TODO: exclude cases with unseen pproc)
  pproc_df = pd.read_csv(pproc_fp)
  dashb_df = dashb_df.join(pproc_df[['PRIMARY_PROC', 'PPROC_DECILE']].set_index('PRIMARY_PROC'), on='PRIMARY_PROC',
                           how='inner')
  print_df_info(dashb_df, dfname='Dashboard df with PProc decile', verbose=verbose)

  # Load CPTs for each case and combine with the existing hierarchy
  cpt_df, cpt_grp_df = pd.read_csv(cpt_fp), pd.read_csv(cpt_grp_fp)
  all_cases_cnt = cpt_df['SURG_CASE_KEY'].nunique()

  # Join with CPT hierarchy group; discard cases if none of their CPTs is present in the existing hierarchy
  cpt_df = cpt_df[cpt_df['CPT_CODE'].apply(lambda x: x.isnumeric())]  # discard rows with non-numeric cpt codes
  cpt_df['CPT_CODE'] = cpt_df['CPT_CODE'].astype(int)
  cpt_df = cpt_df.join(cpt_grp_df.set_index('CPT_CODE'), on='CPT_CODE', how='inner')
  print_df_info(cpt_df, dfname='CPT with Group', verbose=verbose)

  cpt_df = cpt_df.groupby(globals.SURG_CASE_KEY)\
    .agg({
    'CPT_CODE': lambda x: list(x),
    'CPT_GROUP': lambda x: list(x)
  })\
    .reset_index()
  print("\nDiscarded %d cases whose CPT(s) are all unknown!\n" % (all_cases_cnt - cpt_df.shape[0]))

  # Join CPT df with the dashboard df by case key
  dashb_df = dashb_df.join(cpt_df.set_index(globals.SURG_CASE_KEY), on=globals.SURG_CASE_KEY, how='left')\
    .rename(columns={'CPT_CODE': globals.CPTS,
             'CPT_GROUP': 'CPT_GROUPS'})
  dashb_df[globals.CPTS] = dashb_df[globals.CPTS].apply(lambda x: x if isinstance(x, list) else [])
  dashb_df['CPT_GROUPS'] = dashb_df['CPT_GROUPS'].apply(lambda x: x if isinstance(x, list) else [])
  print_df_info(dashb_df, dfname="Dashboard DF with CPT info", verbose=verbose)

  # Join with CCSR df
  diags_df = pd.read_csv(diag_fp)
  print_df_info(diags_df, "Diagnosis DF", other_cols=['ccsr_1'])
  diags_df = diags_df.dropna(axis=0, how='any', subset=['ccsr_1'])\
    .groupby(globals.SURG_CASE_KEY)\
    .agg({'ccsr_1': lambda x: list(x)})\
    .reset_index()\
    .rename(columns={'ccsr_1': globals.CCSRS})

  # Join with dashboard df, and use [] as default value for CCSRS and ICD10S
  dashb_df = dashb_df.join(diags_df.set_index(globals.SURG_CASE_KEY), on=globals.SURG_CASE_KEY, how='left')
  dashb_df[globals.CCSRS] = dashb_df[globals.CCSRS].apply(lambda x: x if isinstance(x, list) else [])
  print_df_info(dashb_df, dfname='Dashboard DF with chronic conditions', verbose=verbose)

  # Remove cases with NaN in feature cols
  dashb_df = dashb_df.dropna(subset=globals.FEATURE_COLS_NO_WEIGHT)
  print_df_info(dashb_df, dfname='Dashboard DF (rows with NA in feature cols are dropped)', verbose=verbose)

  return dashb_df


# helper function that logs the intermediate dataframe info at each filtering step in get_combined_dataframe()
def print_df_info(df, dfname="Dashboard", other_cols=None, verbose=False):
  if verbose:
    print("%s columns:\n" % dfname, df.keys())
    print("Number of cases: %d" % df[globals.SURG_CASE_KEY].nunique())
    if other_cols:
      for col_name in other_cols:
        print("Number of unique, non-NaN values in '%s': %d" % (col_name, df[col_name].nunique(dropna=True)))
    print("Number of NaNs in each column:\n%s" % df.isnull().sum().sort_values(ascending=False))
    print("\n")


# Data preprocessing (clean, discretize and one-hot encode certain features)
def get_feature_matrix(df, skip_cases_fp, cols=globals.FEATURE_COLS_NO_WEIGHT, onehot_cols=None, onehot_dtypes=None,
                       discretize_cols=None):
  # Get target feature names from training data
  target_features = pd.read_csv(skip_cases_fp).columns.to_list()

  # Make data matrix X numeric
  X = df.copy()[cols]
  X.loc[(X.SEX_CODE != 'F'), 'SEX_CODE'] = 0.0
  X.loc[(X.SEX_CODE == 'F'), 'SEX_CODE'] = 1.0

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
        raise NotImplementedError("Cannot encode column '%s' with a data type of '%s'" % (oh_col, dtype))
      X = X.drop(columns=[oh_col]).join(dummies).fillna(0)

    # Drop rows that has certain indicator columns not covered in the target feature list (e.g. an unseen CPT code)
    new_ftrs = set(X.columns.to_list()) - set(target_features) - {globals.SURG_CASE_KEY, 'CPT_GROUPS'}
    if len(new_ftrs) > 0:
      case_idxs_with_new_features = set()
      for new_ftr in new_ftrs:
        idxs = X.index[X[new_ftr] == 1].to_list()
        case_idxs_with_new_features = case_idxs_with_new_features.union(set(idxs))
      print('Dropping %d cases with unseen features' % len(case_idxs_with_new_features))
      print("New features: ", new_ftrs)
      X = X.drop(index=list(case_idxs_with_new_features))\
        .drop(columns=list(new_ftrs))\
        .reset_index(drop=True)
      if X.shape[0] == 0:
        raise Exception("All cases in this dataset contain an unseen indicator feature!")
    # Add uncovered indicators as columns
    uncovered_ftrs = set(target_features) - set(X.columns.to_list())
    X[list(uncovered_ftrs)] = 0.0

  # Save SURG_CASE_KEY, but drop with other non-numeric columns for data matrix
  X_case_key = X[globals.SURG_CASE_KEY].to_numpy()
  X.drop(columns=globals.NON_NUMERIC_COLS, inplace=True, errors='ignore')

  # Reorder columns such that it aligns with target_features
  X = X[target_features]

  # Discretize certain continuous columns by request
  X = X.to_numpy(dtype=np.float64)
  if discretize_cols:
    discretize_columns(X, target_features, discretize_cols, inplace=True)

  # Remove cases identical to any in skip_cases, update X_case_keys correspondingly
  N = X.shape[0]
  X, X_case_key = discard_o2m_cases_from_historical_data(X, X_case_key, skip_cases_fp)
  print('Discarded %d cases with multiple possible outcomes' % (N - X.shape[0]))

  # Basic sanity check
  assert X.shape[1] == len(target_features), 'Generated data matrix has %d features, but feature list has %d items' % \
                                          (X.shape[1], len(target_features))

  return X, target_features, X_case_key


# Return a numpy array of true number of nights, indexed by X_case_keys
def get_outcome_vec(df, X_case_keys):
  return df[df[globals.SURG_CASE_KEY].isin(X_case_keys)][globals.NNT].to_numpy()


# Return a numpy array of SPS-predicted number of nights, indexed by X_case_keys
def get_surgeon_pred_vec(df, X_case_keys):
  return df[df[globals.SURG_CASE_KEY].isin(X_case_keys)][globals.SPS_LOS_FTR].to_numpy()


# Removes cases that historically has multiple possible outcomes
def discard_o2m_cases_from_historical_data(X, X_case_keys, skip_cases_fp):
  # skip_cases_fp: file path of the historical cases that have multiple possible outcomes
  skip_cases_df = pd.read_csv(skip_cases_fp, index_col=False)
  assert skip_cases_df.shape[1] == X.shape[1], "Input data matrix must match with training data in features!"

  Xdf = pd.DataFrame(X, columns=skip_cases_df.columns)
  Xdf[globals.SURG_CASE_KEY] = X_case_keys  # Update the surgical case keys accordingly
  # Left join minus inner join with skip_cases_df
  Xdf = pd.merge(Xdf, skip_cases_df, how='left', on=skip_cases_df.columns.to_list(), indicator=True)
  Xdf = Xdf[Xdf['_merge'] == 'left_only']

  return Xdf[skip_cases_df.columns].to_numpy(), Xdf[globals.SURG_CASE_KEY].to_numpy()


# Discretizes age or weight z-score features
def discretize_columns(X, feature_names, discretize_cols, inplace=False):
  if not inplace:
    X = np.copy(X)
  # Modify data matrix with discretized columns by request
  for dis_col in discretize_cols:
    idx = feature_names.index(dis_col)
    if dis_col == 'AGE_AT_PROC_YRS':
      X[:, idx] = np.digitize(X[:, idx], globals.AGE_BINS)
    elif dis_col == 'WEIGHT_ZSCORE':
      X[:, idx] = np.digitize(X[:, idx], globals.WEIGHT_Z_BINS)
    else:
      raise Warning("%s discretization is not available yet!" % dis_col)
  return X


# Load pre-trained model
def load_pretrained_model(md2model_filename: Dict):
  mds = set(md2model_filename.keys())
  if len(mds - globals.ALL_MODELS) > 0:
    raise NotImplementedError("Input model set contains a model that is not implemented yet!")

  if len(mds) == 1:  # load a single sklearn model
    model_fp = globals.PRETRAINED_MODELS_DIR / list(md2model_filename.values())[0]
    model = joblib.load(model_fp)
  else:  # load an ensemble model
    md2task2clf = defaultdict(lambda: defaultdict(list))
    for md, md_filename in md2model_filename.items():
      md2task2clf[md][globals.TASK_MULTI_CLF] = [joblib.load(globals.PRETRAINED_MODELS_DIR /md_filename)]
    model = ens.Ensemble(tasks=[globals.TASK_MULTI_CLF], md2clfs=md2task2clf)
  return model


# Use pretrained model to predict outcome
def predict_los(X, model):
  return model.predict(X)


# Compute the accuracy with an error tolerance of 1 night
def scorer_1nnt_tol(ytrue, ypred):
  # accuracy within +-1 nnt error tolerance
  acc_1nnt_tol = len(np.where(np.abs(ytrue - ypred) <= 1)[0]) / len(ytrue)
  return acc_1nnt_tol
