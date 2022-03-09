"""
Helper functions to preprocess the data and generate data matrix with its corresponding labels
"""
from IPython.display import display
from imblearn.over_sampling import SMOTENC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from time import time
from typing import Dict

from globals import *
from c1_feature_engineering import FeatureEngineeringModifier, DecileGenerator


def gen_cohort_df(df, cohort):
  if cohort == COHORT_ALL:
    return df
  else:
    ch_pprocs = COHORT_TO_PPROCS[cohort]
    return df.query("PRIMARY_PROC in @ch_pprocs")


class Dataset(object):

  def __init__(self, df, outcome=NNT, ftr_cols=FEATURE_COLS, col2decile_ftrs2aggf=None,
               onehot_cols=[], discretize_cols=None, test_pct=0.2, test_idxs=None, cohort=COHORT_ALL,
               trimmed_ccsr=None, target_features=None, decile_gen=None, remove_o2m=(True, True),
               scaler=None, scale_numeric_only=True):
    # Check args
    assert all(oh in ONEHOT_COL2DTYPE.keys() for oh in onehot_cols), \
      f'onehot_cols must be in {ONEHOT_COL2DTYPE.keys()}!'
    # Define datatype of the columns to be onehot-encoded in order
    onehot_dtypes = [ONEHOT_COL2DTYPE[oh] for oh in onehot_cols]

    self.df = df.copy()
    self.outcome = outcome

    # Filter df by primary procedure cohort
    self.cohort = cohort
    self.cohort_df = gen_cohort_df(df, cohort)
    print('cohort df shape: ', self.cohort_df.shape)

    # 0. Save pure SDA case keys, and those having surgeon prediction case keys
    self.sda_case_keys = self.gen_sda_case_keys(self.cohort_df)
    self.surg_only_case_keys = self.gen_surg_only_case_keys(self.cohort_df)

    # 1. Train-test split
    if test_idxs is None:
      if test_pct == 0:
        train_df, test_df = self.cohort_df, None
      elif test_pct == 1:
        train_df, test_df = None, self.cohort_df
        if col2decile_ftrs2aggf is not None and len(col2decile_ftrs2aggf) > 0:
          assert isinstance(decile_gen, DecileGenerator), "Please specify pre-computed deciles for testing!"
      else:
        train_df, test_df = train_test_split(self.cohort_df, test_size=test_pct)
    else:
      test_df = self.cohort_df.iloc[test_idxs]
      train_df = self.cohort_df.iloc[list(set(range(self.cohort_df.shape[0])) - set(test_idxs))]

    # 2. Initialize required fields
    self.train_df_raw, self.test_df_raw = train_df, test_df
    self.Xtrain, self.ytrain, self.train_case_keys, self.train_cohort_df = None, None, None, None
    self.Xtest, self.ytest, self.test_case_keys, self.test_cohort_df = None, None, None, None
    self.o2m_df_train = None
    self.feature_names = None
    self.FeatureEngineer = FeatureEngineeringModifier(
      onehot_cols, onehot_dtypes, trimmed_ccsr, discretize_cols, col2decile_ftrs2aggf)
    if decile_gen is not None:
      self.FeatureEngineer.set_decile_gen(decile_gen)  # when test_pct = 1

    # 3. Preprocess train & test data
    self.preprocess_train(outcome, ftr_cols, remove_o2m_train=remove_o2m[0])
    self.preprocess_test(outcome, ftr_cols, target_features, remove_o2m_test=remove_o2m[1])

    # 4. Apply data matrix scaling (e.g. normalization, standardization)
    self.input_scaler = scaler
    self.scale_numeric_only = scale_numeric_only
    if self.input_scaler is not None:
      self.scale_Xtrain(how=scaler, only_numeric=self.scale_numeric_only)
      self.scale_Xtest(scaler=self.input_scaler)

  def gen_sda_case_keys(self, df):
    # identify pure SDA cases
    sda_df = df[~(df['SPS_REQUEST_DT_TM'].notnull() & df[SPS_PRED].isnull())]
    return sda_df['SURG_CASE_KEY'].unique().astype(int)

  def gen_surg_only_case_keys(self, df):
    surg_df = df[df[SPS_PRED].notnull()]
    return surg_df['SURG_CASE_KEY'].unique().astype(int)

  def scale_Xtrain(self, how='minmax', only_numeric=True):  # ['minmax', 'std', 'robust']  -- only normalize continuous / ordinal features
    if self.train_df_raw is None or self.Xtrain.shape[0] == 0:
      return self.Xtrain
    if how == 'minmax':
      scaler = MinMaxScaler()
    elif how == 'std':
      scaler = StandardScaler()
    elif how == 'robust':
      scaler = RobustScaler()
    else:
      raise NotImplementedError(f'Scaler {how} is not supported yet!')

    if only_numeric:
      numeric_colidxs = np.where(np.in1d(self.feature_names, ALL_POSSIBLE_NUMERIC_COLS))[0]
      self.Xtrain[:, numeric_colidxs] = scaler.fit_transform(self.Xtrain[:, numeric_colidxs])
    else:
      self.Xtrain = scaler.fit_transform(self.Xtrain)
    self.input_scaler = scaler
    return self.Xtrain

  def scale_Xtest(self, scaler=None):
    if self.test_df_raw is None or self.Xtest.shape[0] == 0:
      return self.Xtest
    if scaler is None:
      scaler = self.input_scaler
    assert scaler is not None, 'Input scaler for Xtest cannot be None!'
    if self.scale_numeric_only:
      numeric_colidxs = np.where(np.in1d(self.feature_names, ALL_POSSIBLE_NUMERIC_COLS))[0]
      self.Xtest[:, numeric_colidxs] = scaler.fit_transform(self.Xtest[:, numeric_colidxs])
    else:
      self.Xtest = scaler.transform(self.Xtest)
    return self.Xtest

  def preprocess_train(self, outcome, ftr_cols=FEATURE_COLS, remove_o2m_train=True):
    print('\n***** Start to preprocess Xtrain:')
    # I. Preprocess training set
    if self.train_df_raw is not None:
      # Preprocess outcome values / SPS prediction in df
      train_df = preprocess_y(self.train_df_raw, outcome, inplace=False)
      if SPS_PRED in train_df.columns:
        train_df = preprocess_y(train_df, outcome, surg_y=True)

      # Modify data matrix
      self.Xtrain, self.feature_names, self.train_case_keys, self.ytrain, self.o2m_df_train = preprocess_Xtrain(
        train_df, outcome, ftr_cols, self.FeatureEngineer, remove_o2m=remove_o2m_train)
      self.train_cohort_df = self.train_df_raw.set_index('SURG_CASE_KEY').loc[self.train_case_keys]
    else:
      self.Xtrain, self.ytrain, self.train_case_keys = np.array([]), np.array([]), np.array([])
      self.train_cohort_df = self.train_df_raw

  def preprocess_test(self, outcome, ftr_cols=FEATURE_COLS, target_features=None, remove_o2m_test=None):
    print('\n***** Start to preprocess Xtest:')
    # II. Preprocess test set, if it's not empty
    if self.test_df_raw is not None:
      # Preprocess ytest & sps prediction
      test_df = preprocess_y(self.test_df_raw, outcome)
      preprocess_y(test_df, outcome, surg_y=True, inplace=True)

      if isinstance(remove_o2m_test, pd.DataFrame):
        o2m_df = remove_o2m_test
        self.feature_names = o2m_df.columns.to_list()  # TODO: refactor this, in case o2m_df is empty
      elif remove_o2m_test == True:
        o2m_df = self.o2m_df_train
      elif remove_o2m_test == False:
        o2m_df = None
        # TODO: why not set feature_names here as well????
      else:
        o2m_df = None
        self.feature_names = target_features
        if self.feature_names is None:
          raise ValueError("target_features cannot be None when test_pct = 1!")

      self.Xtest, _, self.test_case_keys = preprocess_Xtest(test_df, self.feature_names, ftr_cols,
                                                            ftrEng=self.FeatureEngineer, skip_cases_df_or_fp=o2m_df)
      self.test_cohort_df = test_df.set_index('SURG_CASE_KEY').loc[self.test_case_keys]
      self.ytest = np.array(self.test_cohort_df[outcome])
    else:
      self.Xtest, self.ytest, self.test_case_keys = np.array([]), np.array([]), np.array([])
      self.test_cohort_df = self.test_df_raw

  def get_Xytrain_by_case_key(self, query_case_keys):
    if self.train_case_keys is not None:
      idxs = np.where(np.in1d(self.train_case_keys, query_case_keys))[0]
      return self.Xtrain[idxs, :], self.ytrain[idxs]
    return np.array([]), np.array([])

  def get_Xytest_by_case_key(self, query_case_keys):
    if self.test_case_keys is not None:
      idxs = np.where(np.in1d(self.test_case_keys, query_case_keys))[0]
      return self.Xtest[idxs, :], self.ytest[idxs]
    return np.array([]), np.array([])

  def get_sda_Xytrain(self):
    if self.train_case_keys is not None:
      sda_idxs = np.where(np.in1d(self.train_case_keys, self.sda_case_keys))[0]
      return self.Xtrain[sda_idxs, :], self.ytrain[sda_idxs]
    return np.array([]), np.array([])

  def get_sda_Xytest(self):
    if self.test_case_keys is not None:
      sda_idxs = np.where(np.in1d(self.test_case_keys, self.sda_case_keys))[0]
      return self.Xtest[sda_idxs, :], self.ytest[sda_idxs]
    return np.array([]), np.array([])

  def get_cohort_to_Xytrains(self, by_cohort) -> Dict:
    assert by_cohort in {PRIMARY_PROC, SURG_GROUP}
    train_df_by_cohort = self.df.set_index('SURG_CASE_KEY').loc[self.train_case_keys].reset_index() \
      .groupby(by=by_cohort)['SURG_CASE_KEY']
    cohort_to_Xdata = {}
    for cohort, case_keys in train_df_by_cohort:
      idxs = np.where(np.in1d(self.train_case_keys, case_keys))[0]
      cohort_to_Xdata[cohort] = (self.Xtrain[idxs, :], self.ytrain[idxs])
    return cohort_to_Xdata

  def get_cohort_to_Xytests(self, by_cohort):
    assert by_cohort in {PRIMARY_PROC, SURG_GROUP}
    test_df_by_cohort = self.df.set_index('SURG_CASE_KEY').loc[self.test_case_keys].reset_index() \
      .groupby(by=by_cohort)['SURG_CASE_KEY']
    cohort_to_Xdata = {}
    for cohort, case_keys in test_df_by_cohort:
      idxs = np.where(np.in1d(self.test_case_keys, case_keys))[0]
      cohort_to_Xdata[cohort] = (self.Xtest[idxs, :], self.ytest[idxs])
    return cohort_to_Xdata

  def get_surgeon_pred_df_by_case_key(self, query_case_keys):
    if query_case_keys is None or len(query_case_keys) == 0:
      return pd.DataFrame(columns=['SURG_CASE_KEY', SPS_PRED])
    surg_df = self.df[['SURG_CASE_KEY', SPS_PRED, self.outcome, ADMIT_DTM]]\
      .set_index('SURG_CASE_KEY').loc[query_case_keys].reset_index()
    surg_df = surg_df[surg_df[SPS_PRED].notnull()]
    preprocess_y(surg_df, self.outcome, True, inplace=True)  # preprocess surgeon prediction
    preprocess_y(surg_df, self.outcome, False, inplace=True)  # preprocess true outcome
    return surg_df

  def get_year_by_case_key(self, query_case_keys):
    if query_case_keys is None or len(query_case_keys) == 0:
      return pd.DataFrame(columns=['SURG_CASE_KEY', 'YEAR', self.outcome])
    # TODO
    return

  def get_raw_nnt(self, xtype='train'):
    case_keys = self.train_case_keys if xtype.lower() == 'train' else self.test_case_keys
    df = self.df.set_index('SURG_CASE_KEY').loc[case_keys]
    return df[NNT]

  def __str__(self):
    res = "Training set size: %d\n" \
          "Test set size: %d\n"\
          "Number of features: %d\n\n" % (self.ytrain.shape[0], self.ytest.shape[0], len(self.feature_names))
    res += "Feature names: \n" + "\n".join(self.feature_names)
    return res

# TODO: primary cpt
def preprocess_Xtrain(df, outcome, feature_cols, ftrEng: FeatureEngineeringModifier,
                      remove_nonnumeric=True, verbose=False, remove_o2m=True):
  # Make data matrix X numeric
  ftrEng_dep_cols_to_drop = [ftrEng.decile_outcome]  # dependent columns for data cleaning / feature engineering
  if STATE not in feature_cols:
    ftrEng_dep_cols_to_drop.append(STATE)
  Xdf = df.copy()[feature_cols + ftrEng_dep_cols_to_drop]

  # Handle NaNs -- only remove NAs if the nullable column IS IN 'feature_cols'
  Xdf = ftrEng.handle_nans(Xdf)

  # Discard unwanted CCSRs
  Xdf, onehot_cols = ftrEng.trim_ccsr_in_X(Xdf, ftrEng.onehot_cols, ftrEng.trimmed_ccsr)

  # Encode categorical variables
  Xdf = ftrEng.dummy_code_discrete_cols(Xdf)

  # Discretize certain continuous columns by request
  ftrEng.discretize_columns_df(Xdf, ftrEng.discretize_cols, inplace=True)

  # One-hot encode indicator variables
  Xdf = ftrEng.onehot_encode_cols(Xdf, onehot_cols, ftrEng.onehot_dtypes)

  # Generate deciles
  s = time()
  decile_cols = ftrEng.decile_generator.gen_decile_cols(Xdf, ftrEng.decile_outcome, ftrEng.col2decile_ftr2aggf)
  # Generate CPT Group decile if this is not meant to be a feature in Xdata
  cptgrp_decile = ftrEng.decile_generator.gen_cpt_group_decile(Xdf, ftrEng.decile_outcome)
  # if CPT_GROUP not in ftrEng.col2decile_ftr2aggf.keys():
  #   print('generated CPT group decile!')
  ftrEng.decile_generator.cpt_group_decile = cptgrp_decile
  print('Took %d sec to generate all deciles' % (time() - s))

  # Add decile-related columns to Xdf
  Xdf = ftrEng.join_with_all_deciles(Xdf, ftrEng.col2decile_ftr2aggf)
  print("\nAfter adding decile cols: Xdf - ", Xdf.shape)

  # Filter out rare primary procedure cases, and replace them with the CPT group with max decile
  # for ties, simply use 1 of the 7 tie groups
  rare_pproc_to_pr_cptgrp = {}  # rare primary procedure to primary cpt group


  # Save SURG_CASE_KEY, but drop with other non-numeric columns for data matrix
  X_case_keys = Xdf['SURG_CASE_KEY'].to_numpy().astype(int)
  if remove_nonnumeric:
    Xdf.drop(columns=NON_NUMERIC_COLS + ftrEng_dep_cols_to_drop, inplace=True, errors='ignore')
  print("\nAfter droppping nonnumeric cols: Xdf - ", Xdf.shape)
  # Convert dataframe to numerical numpy matrix and save the corresponding features' names
  X = Xdf.to_numpy(dtype=np.float64)
  target_features = Xdf.columns.to_list()

  # Remove cases that have multiple possible outcomes
  o2m_df, y = None, df.set_index('SURG_CASE_KEY').loc[X_case_keys][outcome].to_numpy()
  if remove_o2m:
    s = time()
    o2m_df, X, X_case_keys, y = discard_o2m_cases_from_self(X, X_case_keys, y, target_features)  # training set
    print("\nRemoving o2m cases from self took %d sec" % (time() - s))
    # TODO: ??? regenerate decile if o2m_df.shape[0] > 0
  print('\nAfter removing o2m cases: X - ', X.shape)

  if verbose:
    display(pd.DataFrame(X, columns=target_features).head(20))

  assert len(X_case_keys) == len(np.unique(X_case_keys)), 'Xrain contains duplicated case keys!'
  assert X.shape[1] == len(target_features), 'Generated data matrix has %d features, but feature list has %d items' % \
                                             (X.shape[1], len(target_features))  # Basic sanity check
  return X, target_features, X_case_keys, y, o2m_df


# TODO: primary cpt
# Data preprocessing (clean, discretize and one-hot encode certain features)
def preprocess_Xtest(df, target_features, feature_cols, ftrEng: FeatureEngineeringModifier, skip_cases_df_or_fp=None):
  # Get target feature names from training data
  if skip_cases_df_or_fp is None:
    skip_cases_df = None
  elif isinstance(skip_cases_df_or_fp, pd.DataFrame):
    skip_cases_df = skip_cases_df_or_fp
  else:
    skip_cases_df = pd.read_csv(skip_cases_df_or_fp)

  Xdf = df.copy()[feature_cols]

  # Handle NAs
  Xdf = ftrEng.handle_nans(Xdf, isTrain=False)

  # Add a column of trimmed CCSRs with/without a column of the corresponding trimmed ICD10s
  Xdf, onehot_cols = ftrEng.trim_ccsr_in_X(Xdf, ftrEng.onehot_cols, ftrEng.trimmed_ccsr)

  # Make data matrix X numeric
  Xdf = ftrEng.dummy_code_discrete_cols(Xdf)

  # Discretize certain continuous columns by request
  ftrEng.discretize_columns_df(Xdf, ftrEng.discretize_cols, inplace=True)

  if onehot_cols is not None:
    # One-hot encode the required columns according to a given historical set of features (e.g. CPT, CCSR etc.)
    Xdf = ftrEng.onehot_encode_cols(Xdf, onehot_cols, ftrEng.onehot_dtypes)

    # Match Xdf to the target features from the training data matrix (for now, only medical codes lead to such need)
    Xdf = ftrEng.match_Xdf_cols_to_target_features(Xdf, target_features)

  # Join with existing deciles (do this step before match_Xdf_cols_to_target_features())
  Xdf = ftrEng.join_with_all_deciles(Xdf, ftrEng.col2decile_ftr2aggf)

  # TODO: Scan for uniform columns

  # Save SURG_CASE_KEY, but drop with other non-numeric columns for data matrix
  X_case_keys = Xdf['SURG_CASE_KEY'].to_numpy().astype(int)
  Xdf.drop(columns=NON_NUMERIC_COLS, inplace=True, errors='ignore')

  # Reorder columns such that it aligns with target_features
  Xdf = Xdf[target_features]

  # Convert feature-engineered data matrix to numerical numpy array
  X = Xdf.to_numpy(dtype=np.float64)

  # Remove cases identical to any in skip_cases, update X_case_keys correspondingly
  if (skip_cases_df is not None) and (not skip_cases_df.empty):
    s = time()
    X, X_case_keys = discard_o2m_cases_from_historical_data(X, X_case_keys, skip_cases_df)
    print("Removing o2m cases according to training set took %d sec" % (time() - s))

  # Basic sanity check
  assert len(X_case_keys) == len(np.unique(X_case_keys)), 'Xtest contains duplicated case keys!'
  assert X.shape[1] == len(target_features), 'Generated data matrix has %d features, but feature list has %d items' % \
                                             (X.shape[1], len(target_features))
  return X, target_features, X_case_keys


def get_df_by_case_keys(df, case_keys):
  return df.set_index('SURG_CASE_KEY').loc[case_keys].reset_index()


def gen_rare_pproc_Xdf(Xdf: pd.DataFrame, min_train_cohort_size=40):
  pproc_size = Xdf.groupby(PRIMARY_PROC).size().reset_index(name='pproc_size')
  rare_pproc_df = Xdf.join(pproc_size.set_index(PRIMARY_PROC), on=PRIMARY_PROC, how='left')
  rare_pproc_df = rare_pproc_df[rare_pproc_df['pproc_size'] < min_train_cohort_size]
  return rare_pproc_df


def replace_pproc_with_primary_cptgrp(rare_Xdf: pd.DataFrame, cptgrp_decile: pd.DataFrame):
  # explode on cptgroups, join with cptgrp decile, groupby surg key & agg on max without tie breaking
  # if only 1 max decile, use the corresponding cptgroup, else replace with 'CPTGROUP_TIE?' ? = max decile
  print('[pproc <-- pr_cptgrp] Input Xdf case count: ', rare_Xdf['SURG_CASE_KEY'].nunique())
  rare_Xdf_exp = rare_Xdf[['SURG_CASE_KEY', CPT_GROUPS, PRIMARY_PROC]].explode(CPT_GROUPS)\
    .rename(columns={CPT_GROUPS: CPT_GROUP})  # shouldn't contain na after explosion
  prev_exp_N = rare_Xdf_exp.shape[0]
  rare_Xdf_exp = rare_Xdf_exp.dropna(subset=[CPT_GROUP])
  assert prev_exp_N == rare_Xdf_exp.shape[0], 'Certain cases have CPT_GROUPS = [] !'

  # Generate all max CPT group deciles for the exploded Xdf
  rare_Xdf_cptgrp_decile = rare_Xdf_exp.join(cptgrp_decile.set_index(CPT_GROUP), on=CPT_GROUP, how='inner').reset_index(drop=True)
  max_mask = rare_Xdf_cptgrp_decile.groupby('SURG_CASE_KEY')[CPT_GROUP_DECILE].apply(lambda x: x == x.max())  # do not break tie
  MAX_CPTGRP_DECILE = 'MAX_' + CPT_GROUP_DECILE
  MAX_CPTGRP_DECILE_CNT = MAX_CPTGRP_DECILE + '_COUNT'
  rare_Xdf_max_cptgrp_decile = rare_Xdf_cptgrp_decile[max_mask].rename(columns={CPT_GROUP_DECILE: MAX_CPTGRP_DECILE})

  # Count the number of max CPT group decile for each case
  max_decile_cnt = rare_Xdf_max_cptgrp_decile \
    .groupby('SURG_CASE_KEY')[MAX_CPTGRP_DECILE].count() \
    .reset_index(name=MAX_CPTGRP_DECILE_CNT) \
    .set_index('SURG_CASE_KEY')

  # Cases with a single cptgrp decile max -- primary_proc_cptgrp = cptgrp
  single_max_cptgrp_decile_Xdf = rare_Xdf_max_cptgrp_decile.join(
    max_decile_cnt[max_decile_cnt[MAX_CPTGRP_DECILE_CNT] == 1], on='SURG_CASE_KEY', how='inner')
  single_max_cptgrp_decile_Xdf[PRIMARY_PROC_CPTGRP] = single_max_cptgrp_decile_Xdf[CPT_GROUP]

  # Cases with multiple cptgrp decile max -- primary_proc_cptgrp = pr_cptgrp_x (x = cptgrp_decile)
  multi_max_cptgrp_decile_Xdf = rare_Xdf_max_cptgrp_decile.join(
    max_decile_cnt[max_decile_cnt[MAX_CPTGRP_DECILE_CNT] > 1], on='SURG_CASE_KEY', how='inner') \
    .drop_duplicates(subset=['SURG_CASE_KEY', MAX_CPTGRP_DECILE_CNT], keep='first')
  multi_max_cptgrp_decile_Xdf[PRIMARY_PROC_CPTGRP] = multi_max_cptgrp_decile_Xdf[MAX_CPTGRP_DECILE] \
    .apply(lambda x: PR_CPTGRP_X + str(x))

  return single_max_cptgrp_decile_Xdf, multi_max_cptgrp_decile_Xdf

# TODO: what if 1 pproc is mapped to multiple primary cpt groups?? -- double check, acceptable I think
# TODO: when processing Xtest, need to filter out unseen Primary CPTGRP??


def update_rare_pproc_with_primary_cptgroup(Xdf: pd.DataFrame):
  # encompass the above 2 funcs, join case w/ rare pproc with the original Xdf

  return
# TODO: how to use this? add an arg (e.g. replace_rare_pproc?) in Dataset init?


def preprocess_y(df: pd.DataFrame, outcome, surg_y=False, inplace=False):
  # Generate an outcome vector y with shape: (n_samples, )
  df = df.copy() if not inplace else df

  if outcome == LOS:
    return df.LENGTH_OF_STAY.to_numpy() if not surg_y else df[SPS_PRED].to_numpy()
  elif outcome == ">12h":
    y = df.LENGTH_OF_STAY.to_numpy() if not surg_y else df[SPS_PRED].to_numpy()
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
  elif outcome == ">1d":
    y = df.LENGTH_OF_STAY.to_numpy() if not surg_y else df[SPS_PRED].to_numpy()
    y[y > 1] = 1
    y[y <= 1] = 0
  elif outcome == NNT:
    y = df[NNT].to_numpy() if not surg_y else df[SPS_PRED].to_numpy()
    y = gen_y_nnt(y)
  elif outcome.endswith("nnt"):
    y = df[NNT].to_numpy() if not surg_y else df[SPS_PRED].to_numpy()
    cutoff = int(outcome.split("nnt")[0])
    y = gen_y_nnt_binary(y, cutoff)
  else:
    raise NotImplementedError("Outcome type '%s' is not implemented yet!" % outcome)

  # Update df by adding or updating the outcome column
  if surg_y:
    df[SPS_PRED] = y
  else:
    df[outcome] = y
  return df


def gen_y_nnt(y):
  if isinstance(y, pd.Series):
    y = y.to_numpy()
  elif not isinstance(y, np.ndarray):
    y = np.array(y)
  y = np.round(y)
  y[y > MAX_NNT] = MAX_NNT + 1
  return y


def gen_y_nnt_binary(y, cutoff):  # task: predict if LoS <= cutoff (cutoff in range(0, MAX_NNT + 1)
  yb = np.copy(y)
  yb[y <= cutoff] = 1
  yb[y > cutoff] = 0
  return yb


# Perform train-test split
def gen_train_test(X, y, test_pct=0.2, rand_state=SEED):
  """
  Returns the purely integer-location based index w.r.t. the data matrix X for the train and test set respectively.
  i.e. max(X_train.index) <= n_samples - 1 and max(X_test.index) <= n_samples - 1
  These index lists are not the row index label of the original df, because X is already a numeric matrix and
  has lost the index labels during the conversion.

  :param X: a numeric matrix (n_samples, n_features)
  :param y: a response vector (n_samples, )
  :param test_pct: desired test set percentage, a float between 0 and 1
  :param rand_state: seed for the random split

  :return: training and test set data matrix, response vector and location-based index w.r.t. X
  """
  assert isinstance(X, pd.DataFrame), "Input data needs to be a DateFrame object"
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, random_state=rand_state)

  return X_train, X_test, y_train, y_test, X_train.index.to_numpy(), X_test.index.to_numpy()


# train-validation-test split
def gen_train_val_test(X, y, val_pct=0.2, test_pct=0.2):
  X, y = pd.DataFrame(X), pd.Series(y)
  X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_pct, random_state=SEED)
  X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_pct, random_state=SEED)
  return X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy(),\
         X_train.index, X_val.index, X_test.index


def standardize(X_train, X_test=None):
  scaler = StandardScaler().fit(X_train)
  if X_test is None:
    return scaler.transform(X_train)
  return scaler.transform(X_test)


def gen_smote_Xy(X, y, feature_names):
  categ_ftrs = np.array([False if ftr in CONTINUOUS_COLS else True for ftr in feature_names], dtype=bool)
  sm = SMOTENC(categorical_features=categ_ftrs, random_state=SEED)
  X, y = sm.fit_resample(X, y)
  return X, y


# Removes cases that historically has multiple possible outcomes
def discard_o2m_cases_from_historical_data(X, X_case_keys, skip_cases_df):
  # skip_cases_df: df of the historical cases that have multiple possible outcomes
  skip_cases_df = pd.DataFrame(skip_cases_df)
  assert skip_cases_df.shape[1] == X.shape[1], "Input data matrix must match with training data in features!"

  Xdf = pd.DataFrame(X, columns=skip_cases_df.columns)
  Xdf['SURG_CASE_KEY'] = X_case_keys  # Update the surgical case keys accordingly

  # Left join minus inner join with skip_cases_df
  Xdf = pd.merge(Xdf, skip_cases_df, how='left', on=skip_cases_df.columns.to_list(), indicator=True)
  Xdf = Xdf[Xdf['_merge'] == 'left_only']
  print("Skipped %d cases" % (len(X_case_keys) - Xdf.shape[0]))

  return Xdf[skip_cases_df.columns].to_numpy(), Xdf['SURG_CASE_KEY'].to_numpy()


# Removes cases that has multiple possible outcomes from input data
def discard_o2m_cases_from_self(X, X_case_keys, y, features):
  o2m_df = gen_o2m_cases(X, y, features, unique=False, X_case_keys=X_case_keys)
  o2m_idx = o2m_df.index.to_list()
  #print("One-to-many cases index: ", o2m_idx)
  print("#O2M cases: ", len(o2m_idx))
  keep_idxs = np.delete(np.arange(X.shape[0]), o2m_idx)
  X, X_case_keys, y = X[keep_idxs, :], X_case_keys[keep_idxs], y[keep_idxs]
  return o2m_df.drop_duplicates(subset=features), X, X_case_keys, y


# Generate a list of (unique) cases that have at least one identical case with a different outcome
# TODO: why add X_case_keys? -- output these O2M cases for cherrypick when unique = False
def gen_o2m_cases(X, y, features, unique=True, save_fp=None, X_case_keys=None):
  Xydf = pd.DataFrame(X, columns=features)
  Xydf['Outcome'] = y
  if X_case_keys is not None:
    assert len(X_case_keys) == Xydf.shape[0], '[gen_o2m_cases] X_case_keys must align with X, y'
    Xydf.insert(loc=0, column='SURG_CASE_KEY', value=X_case_keys)
  dup_mask = Xydf.duplicated(subset=features, keep=False)
  Xydf_dup = Xydf[dup_mask]

  # Get cases with identical features but different outcomes
  o2m_df = Xydf_dup \
    .groupby(by=features)\
    .filter(lambda x: len(x['Outcome'].value_counts().index) > 1)
  # o2m_df.groupby(by=features)\
  #   .agg(lambda x: list(x))\
  #   .to_csv('featureVector_to_o2m_case_keys2.csv', index=False)
  print("Covered %d o2m cases" % o2m_df.shape[0])

  if unique:
    o2m_df = o2m_df.drop_duplicates(subset=features)
    print("Contains %d unique o2m cases" % o2m_df.shape[0])
  o2m_df.drop(columns=['Outcome', 'SURG_CASE_KEY'], inplace=True, errors='ignore')

  if save_fp:
    o2m_df.to_csv(save_fp, index=False)
  return o2m_df


# Get X, y for pure SDA cases and/or cases with a surgeon prediction
def get_Xys_sda_surg(dataset: Dataset, sda_only, surg_only):
  if sda_only:
    query_case_keys = dataset.sda_case_keys
    if surg_only:
      query_case_keys = np.intersect1d(query_case_keys, dataset.surg_only_case_keys)
    Xtrain, ytrain = dataset.get_Xytrain_by_case_key(query_case_keys)
    Xtest, ytest = dataset.get_Xytest_by_case_key(query_case_keys)
  else:
    if surg_only:
      Xtrain, ytrain = dataset.get_Xytrain_by_case_key(dataset.surg_only_case_keys)
      Xtest, ytest = dataset.get_Xytest_by_case_key(dataset.surg_only_case_keys)
    else:
      Xtrain, ytrain = dataset.Xtrain, dataset.ytrain
      Xtest, ytest = dataset.Xtest, dataset.ytest
  return Xtrain, ytrain, Xtest, ytest
