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

import globals
from c7_feature_engineering import FeatureEngineeringModifier, DecileGenerator


def gen_cohort_df(df, cohort):
  if cohort == globals.COHORT_ALL:
    return df
  else:
    ch_pprocs = globals.COHORT_TO_PPROCS[cohort]
    return df.query("PRIMARY_PROC in @ch_pprocs")


class Dataset(object):

  def __init__(self, df, outcome=globals.NNT, ftr_cols=globals.FEATURE_COLS, col2decile_ftrs2aggf=None,
               onehot_cols=[], discretize_cols=None, test_pct=0.2, test_idxs=None, cohort=globals.COHORT_ALL,
               trimmed_ccsr=None, target_features=None, decile_gen=None, remove_o2m=(True, True),
               scaler=None, scale_numeric_only=True):
    # Check args
    assert all(oh in globals.ONEHOT_COL2DTYPE.keys() for oh in onehot_cols), \
      f'onehot_cols must be in {globals.ONEHOT_COL2DTYPE.keys()}!'
    # Define datatype of the columns to be onehot-encoded in order
    onehot_dtypes = [globals.ONEHOT_COL2DTYPE[oh] for oh in onehot_cols]

    self.df = df.copy()
    self.outcome = outcome

    # Filter df by primary procedure cohort
    self.cohort = cohort
    self.cohort_df = gen_cohort_df(df, cohort)
    print('cohort df shape: ', self.cohort_df.shape)

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
      numeric_colidxs = np.where(np.in1d(self.feature_names, globals.ALL_POSSIBLE_NUMERIC_COLS))[0]
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
      numeric_colidxs = np.where(np.in1d(self.feature_names, globals.ALL_POSSIBLE_NUMERIC_COLS))[0]
      self.Xtest[:, numeric_colidxs] = scaler.fit_transform(self.Xtest[:, numeric_colidxs])
    else:
      self.Xtest = scaler.transform(self.Xtest)
    return self.Xtest

  def preprocess_train(self, outcome, ftr_cols=globals.FEATURE_COLS, remove_o2m_train=True):
    print('\n***** Start to preprocess Xtrain:')
    # I. Preprocess training set
    if self.train_df_raw is not None:
      # Preprocess outcome values / SPS prediction in df
      train_df = preprocess_y(self.train_df_raw, outcome, inplace=False)
      if globals.SPS_PRED in train_df.columns:
        train_df = preprocess_y(train_df, outcome, sps_y=True)

      # Modify data matrix
      self.Xtrain, self.feature_names, self.train_case_keys, self.ytrain, self.o2m_df_train = preprocess_Xtrain(
        train_df, outcome, ftr_cols, self.FeatureEngineer, remove_o2m=remove_o2m_train)
      self.train_cohort_df = self.train_df_raw.set_index('SURG_CASE_KEY').loc[self.train_case_keys]
    else:
      self.Xtrain, self.ytrain, self.train_case_keys = np.array([]), np.array([]), np.array([])
      self.train_cohort_df = self.train_df_raw

  def preprocess_test(self, outcome, ftr_cols=globals.FEATURE_COLS, target_features=None, remove_o2m_test=None):
    print('\n***** Start to preprocess Xtest:')
    # II. Preprocess test set, if it's not empty
    if self.test_df_raw is not None:
      # Preprocess ytest & sps prediction
      test_df = preprocess_y(self.test_df_raw, outcome)
      preprocess_y(test_df, outcome, sps_y=True, inplace=True)

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
      self.ytest = self.test_cohort_df[outcome]
    else:
      self.Xtest, self.ytest, self.test_case_keys = np.array([]), np.array([]), np.array([])
      self.test_cohort_df = self.test_df_raw

  def get_Xtrain_by_case_key(self, query_case_keys):
    idxs = np.where(np.in1d(self.train_case_keys, query_case_keys))[0]
    return self.Xtrain[idxs, :]

  def get_Xtest_by_case_key(self, query_case_keys):
    idxs = np.where(np.in1d(self.test_case_keys, query_case_keys))[0]
    return self.Xtest[idxs, :]

  def get_surgeon_pred_df_by_case_key(self, query_case_keys):
    if query_case_keys is None or len(query_case_keys) == 0:
      return pd.DataFrame(columns=['SURG_CASE_KEY', globals.SPS_PRED])
    surg_df = self.df[['SURG_CASE_KEY', globals.SPS_PRED, self.outcome]]\
      .set_index('SURG_CASE_KEY').loc[query_case_keys].reset_index()
    surg_df = surg_df[surg_df[globals.SPS_PRED].notnull()]
    preprocess_y(surg_df, self.outcome, True, inplace=True)
    preprocess_y(surg_df, self.outcome, False, inplace=True)
    return surg_df

  def get_raw_nnt(self, xtype='train'):
    case_keys = self.train_case_keys if xtype.lower() == 'train' else self.test_case_keys
    df = self.df.set_index('SURG_CASE_KEY').loc[case_keys]
    return df[globals.NNT]

  def __str__(self):
    res = "Training set size: %d\n" \
          "Test set size: %d\n"\
          "Number of features: %d\n\n" % (self.ytrain.shape[0], self.ytest.shape[0], len(self.feature_names))
    res += "Feature names: \n" + "\n".join(self.feature_names)
    return res


def preprocess_Xtrain(df, outcome, feature_cols, ftrEng: FeatureEngineeringModifier,
                      remove_nonnumeric=True, verbose=False, remove_o2m=True):
  # Make data matrix X numeric
  Xdf = df.copy()[feature_cols + [ftrEng.decile_outcome]]  # add outcome col for computing decile features

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
  print('Took %d sec to generate all deciles' % (time() - s))

  # Add decile-related columns to Xdf
  Xdf = ftrEng.join_with_all_deciles(Xdf, ftrEng.col2decile_ftr2aggf)
  print("\nAfter adding decile cols: Xdf - ", Xdf.shape)

  # Save SURG_CASE_KEY, but drop with other non-numeric columns for data matrix
  X_case_keys = Xdf['SURG_CASE_KEY'].to_numpy()
  if remove_nonnumeric:
    Xdf.drop(columns=globals.NON_NUMERIC_COLS + [ftrEng.decile_outcome], inplace=True, errors='ignore')
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
  assert X.shape[1] == len(target_features), 'Generated data matrix has %d features, but feature list has %d items' % \
                                             (X.shape[1], len(target_features))  # Basic sanity check
  return X, target_features, X_case_keys, y, o2m_df


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
  Xdf = ftrEng.handle_nans(Xdf)

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
  X_case_key = Xdf['SURG_CASE_KEY'].to_numpy()
  Xdf.drop(columns=globals.NON_NUMERIC_COLS, inplace=True, errors='ignore')

  # Reorder columns such that it aligns with target_features
  Xdf = Xdf[target_features]

  # Convert feature-engineered data matrix to numerical numpy array
  X = Xdf.to_numpy(dtype=np.float64)

  # Remove cases identical to any in skip_cases, update X_case_keys correspondingly
  if (skip_cases_df is not None) and (not skip_cases_df.empty):
    s = time()
    X, X_case_key = discard_o2m_cases_from_historical_data(X, X_case_key, skip_cases_df)
    print("Removing o2m cases according to training set took %d sec" % (time() - s))

  # Basic sanity check
  assert X.shape[1] == len(target_features), 'Generated data matrix has %d features, but feature list has %d items' % \
                                             (X.shape[1], len(target_features))
  return X, target_features, X_case_key


def get_df_by_case_keys(df, case_keys):
  return df.set_index('SURG_CASE_KEY').loc[case_keys].reset_index()


def preprocess_y(df: pd.DataFrame, outcome, sps_y=False, inplace=False):
  # Generate an outcome vector y with shape: (n_samples, )
  df = df.copy() if not inplace else df

  if outcome == globals.LOS:
    return df.LENGTH_OF_STAY.to_numpy() if not sps_y else df[globals.SPS_PRED].to_numpy()
  elif outcome == ">12h":
    y = df.LENGTH_OF_STAY.to_numpy() if not sps_y else df[globals.SPS_PRED].to_numpy()
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
  elif outcome == ">1d":
    y = df.LENGTH_OF_STAY.to_numpy() if not sps_y else df[globals.SPS_PRED].to_numpy()
    y[y > 1] = 1
    y[y <= 1] = 0
  elif outcome == globals.NNT:
    y = df[globals.NNT].to_numpy() if not sps_y else df[globals.SPS_PRED].to_numpy()
    y = gen_y_nnt(y)
  elif outcome.endswith("nnt"):
    y = df[globals.NNT].to_numpy() if not sps_y else df[globals.SPS_PRED].to_numpy()
    cutoff = int(outcome.split("nnt")[0])
    y = gen_y_nnt_binary(y, cutoff)
  else:
    raise NotImplementedError("Outcome type '%s' is not implemented yet!" % outcome)

  # Update df by adding or updating the outcome column
  if sps_y:
    df[globals.SPS_PRED] = y
  else:
    df[outcome] = y
  return df


def gen_y_nnt(y):
  if isinstance(y, pd.Series):
    y = y.to_numpy()
  elif not isinstance(y, np.ndarray):
    y = np.array(y)
  y = np.round(y)
  y[y > globals.MAX_NNT] = globals.MAX_NNT + 1
  return y


def gen_y_nnt_binary(y, cutoff):  # task: predict if LoS <= cutoff (cutoff in range(0, MAX_NNT + 1)
  yb = np.copy(y)
  yb[y <= cutoff] = 1
  yb[y > cutoff] = 0
  return yb


# Perform train-test split
def gen_train_test(X, y, test_pct=0.2, rand_state=globals.SEED):
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
  X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_pct, random_state=globals.SEED)
  X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_pct, random_state=globals.SEED)
  return X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy(),\
         X_train.index, X_val.index, X_test.index


def standardize(X_train, X_test=None):
  scaler = StandardScaler().fit(X_train)
  if X_test is None:
    return scaler.transform(X_train)
  return scaler.transform(X_test)


def gen_smote_Xy(X, y, feature_names):
  categ_ftrs = np.array([False if ftr in globals.CONTINUOUS_COLS else True for ftr in feature_names], dtype=bool)
  sm = SMOTENC(categorical_features=categ_ftrs, random_state=globals.SEED)
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
