"""
Helper functions to preprocess the data and generate data matrix with its corresponding labels
"""
from collections import Counter

from IPython.display import display
from imblearn.over_sampling import SMOTENC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time

from . import globals
from . import c0_data_prepare as dp
from .c7_feature_engineering import FeatureEngineeringModifier, DecileGenerator


def gen_cohort_df(df, cohort):
  if cohort == globals.COHORT_ALL:
    return df
  elif cohort == globals.COHORT_ORTHO:  # TODO: double check if this exactly captures orthopedics cohort
    return df[df['SURG_GROUP'] == 'ORTHO OR']
  else:
    ch_pprocs = globals.COHORT_TO_PPROCS[cohort]
    return df.query("PRIMARY_PROC in @ch_pprocs")


class Dataset(object):

  def __init__(self, df, outcome=globals.NNT, ftr_cols=globals.FEATURE_COLS, col2decile_ftrs2aggf=None,
               onehot_cols=None, onehot_dtypes=None, decile_gen=None,
               test_pct=0.2, test_idxs=None, discretize_cols=None,
               cohort=globals.COHORT_ALL, trimmed_ccsr=None, target_features=None, remove_o2m=(True, True)):
    # Check args
    if (onehot_cols is not None) and (onehot_dtypes is not None):
      assert len(onehot_cols) == len(onehot_dtypes), "One-hot Encoding columns and dtypes must match!"

    self.df = df.copy()

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
      self.FeatureEngineer.set_decile_gen(decile_gen)

    # 3. Preprocess train & test data
    self.preprocess_train_test(outcome, ftr_cols, target_features, remove_o2m)

  def preprocess_train_test(self, outcome, ftr_cols=globals.FEATURE_COLS, target_features=None, remove_o2m=(None, None)):

    # I. Preprocess training set
    if self.train_df_raw is not None:
      # Preprocess outcome values / SPS prediction in df
      train_df = preprocess_y(self.train_df_raw, outcome)
      if globals.SPS_LOS_FTR in train_df.columns.to_list():
        train_df = preprocess_y(train_df, outcome, sps_y=True)

      # Modify data matrix
      self.Xtrain, self.feature_names, self.train_case_keys, self.ytrain, self.o2m_df_train = preprocess_Xtrain(
        train_df, outcome, ftr_cols, self.FeatureEngineer, remove_o2m=remove_o2m[0])
      self.train_cohort_df = self.train_df_raw[self.train_df_raw['SURG_CASE_KEY'].isin(self.train_case_keys)]
    else:
      self.Xtrain, self.ytrain, self.train_case_keys = np.array([]), np.array([]), np.array([])
      self.train_cohort_df = self.train_df_raw

    # II. Preprocess test set, if it's not empty
    if self.test_df_raw is not None:
      if isinstance(remove_o2m[1], pd.DataFrame):
        o2m_df = remove_o2m[1]
        self.feature_names = o2m_df.columns.to_list()  # TODO: refactor this, in case o2m_df is empty
      elif remove_o2m[1] == True:
        o2m_df = self.o2m_df_train
      elif remove_o2m[1] == False:
        o2m_df = None
        # TODO: why not set feature_names here as well????
      else:
        o2m_df = None
        self.feature_names = target_features
        if self.feature_names is None:
          raise ValueError("target_features cannot be None when test_pct = 1!")

      self.Xtest, _, self.test_case_keys = preprocess_Xtest(self.test_df_raw, self.feature_names, ftr_cols,
                                                            ftrEng=self.FeatureEngineer, skip_cases_df_or_fp=o2m_df)
      self.ytest = self.test_df_raw[self.test_df_raw['SURG_CASE_KEY'].isin(self.test_case_keys)][outcome].to_numpy()
      self.test_cohort_df = self.test_df_raw[self.test_df_raw['SURG_CASE_KEY'].isin(self.test_case_keys)]
    else:
      self.Xtest, self.ytest, self.test_case_keys = np.array([]), np.array([]), np.array([])
      self.test_cohort_df = self.test_df_raw

  def get_Xtrain_by_case_key(self, query_case_keys):
    idxs = np.where(np.in1d(self.train_case_keys, query_case_keys))[0]
    return self.Xtrain[idxs, :]

  def get_Xtest_by_case_key(self, query_case_keys):
    idxs = np.where(np.in1d(self.test_case_keys, query_case_keys))[0]
    return self.Xtest[idxs, :]

  def __str__(self):
    res = "Training set size: %d\n" \
          "Test set size: %d\n"\
          "Number of features: %d\n\n" % (self.ytrain.shape[0], self.ytest.shape[0], len(self.feature_names))
    res += "Feature names: \n" + "\n".join(self.feature_names)
    return res


def preprocess_Xtrain(df, outcome, feature_cols, ftrEng: FeatureEngineeringModifier,
                      remove_nonnumeric=True, verbose=False, remove_o2m=True):
  # Make data matrix X numeric
  Xdf = df.copy()[feature_cols+[ftrEng.decile_outcome]]

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
  print(Xdf.columns)
  Xdf = ftrEng.join_with_all_deciles(Xdf, ftrEng.col2decile_ftr2aggf)

  # Save SURG_CASE_KEY, but drop with other non-numeric columns for data matrix
  X_case_keys = Xdf['SURG_CASE_KEY'].to_numpy()
  if remove_nonnumeric:
    Xdf.drop(columns=globals.NON_NUMERIC_COLS+[ftrEng.decile_outcome], inplace=True, errors='ignore')

  # Convert dataframe to numerical numpy matrix and save the corresponding features' names
  X = Xdf.to_numpy(dtype=np.float64)
  target_features = Xdf.columns.to_list()

  # Remove cases that have multiple possible outcomes
  o2m_df, y = None, df[outcome].to_numpy()
  if remove_o2m:
    s = time()
    o2m_df, X, X_case_keys, y = discard_o2m_cases_from_self(X, X_case_keys, y, target_features)  # training set
    print("Removing o2m cases from self took %d sec" % (time() - s))

    # TODO: regenerate decile if o2m_df.shape[0] > 0


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

  # Join with existing deciles -- TODO: Test this
  Xdf = ftrEng.join_with_all_deciles(Xdf, ftrEng.col2decile_ftr2aggf)

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
  return df[df['SURG_CASE_KEY'].isin(case_keys)]


def preprocess_y(df: pd.DataFrame, outcome, sps_y=False, inplace=False):
  # Generate an outcome vector y with shape: (n_samples, )
  df = df.copy() if not inplace else df
  y = np.array(df.LENGTH_OF_STAY.to_numpy())
  if outcome == globals.LOS:
    return y
  elif outcome == ">12h":
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
  elif outcome == ">1d":
    y[y > 1] = 1
    y[y <= 1] = 0
  elif outcome == globals.NNT:
    y = gen_y_nnt(df[globals.NNT])
  elif outcome.endswith("nnt"):
    cutoff = int(outcome.split("nnt")[0])
    y = gen_y_nnt_binary(y, cutoff)
  else:
    raise NotImplementedError("Outcome type '%s' is not implemented yet!" % outcome)

  # Update df by adding the outcome column
  if sps_y:
    df['SPS_PREDICTED_LOS'] = y
  else:
    df[outcome] = y
  return df


def gen_y_nnt(dfcol):
  y = np.array(dfcol.to_numpy())
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
  idxs = np.delete(np.arange(X.shape[0]), o2m_idx)
  X, X_case_keys, y = X[idxs, :], X_case_keys[idxs], y[idxs]
  return o2m_df.drop_duplicates(subset=features), X, X_case_keys, y


# Generate a list of (unique) cases that have at least one identical case with a different outcome
# TODO: why add X_case_keys? -- output these O2M cases for cherrypick when unique = False
def gen_o2m_cases(X, y, features, unique=True, save_fp=None, X_case_keys=None):
  Xydf = pd.DataFrame(X, columns=features)
  Xydf['Outcome'] = y
  if X_case_keys is not None:
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

#
# def dummy_code_discrete_cols(Xdf):
#   # Gender
#   Xdf.loc[(Xdf.SEX_CODE != 'F'), 'SEX_CODE'] = 0.0
#   Xdf.loc[(Xdf.SEX_CODE == 'F'), 'SEX_CODE'] = 1.0
#
#   Xdf_cols = Xdf.columns.to_list()
#   # Interpreter need or not
#   if 'INTERPRETER_NEED' in Xdf_cols:
#     Xdf.loc[(Xdf.INTERPRETER_NEED == 'N'), 'INTERPRETER_NEED'] = 0.0
#     Xdf.loc[(Xdf.INTERPRETER_NEED == 'Y'), 'INTERPRETER_NEED'] = 1.0
#
#   # State code
#   if 'STATE_CODE' in Xdf_cols:
#     Xdf[['IN_STATE', 'OUT_OF_STATE_US', 'FOREIGN']] = 0.0
#     Xdf.loc[(Xdf.STATE_CODE == 'MA'), 'IN_STATE'] = 1.0
#     Xdf.loc[(Xdf.STATE_CODE == 'Foreign'), 'FOREIGN'] = 1.0
#     Xdf.loc[((Xdf.IN_STATE == 0.0) & (Xdf.FOREIGN == 0.0)), 'OUT_OF_STATE_US'] = 1.0
#
#   # Language   TODO: Unable to Collect == all 0s or a separate col or discard??
#   if 'LANGUAGE_DESC' in Xdf_cols:
#     Xdf[['ENGLISH', 'SPANISH', 'OTHER_LANGUAGE']] = 0.0
#     Xdf.loc[(Xdf.LANGUAGE_DESC == 'English'), 'ENGLISH'] = 1.0
#     Xdf.loc[(Xdf.LANGUAGE_DESC == 'Spanish'), 'SPANISH'] = 1.0
#     Xdf.loc[(Xdf.LANGUAGE_DESC == 'Unable to Collect'), 'UNKNOWN_LANGUAGE'] = 1.0
#     Xdf.loc[((Xdf.ENGLISH == 0.0) & (Xdf.SPANISH == 0.0) & (Xdf.UNKNOWN_LANGUAGE == 0.0)), 'OTHER_LANGUAGE'] = 1.0
#
#     Xdf.drop(columns=['STATE_CODE', 'LANGUAGE_DESC'], inplace=True)
#   return Xdf
#
#
# def trim_ccsr_in_X(Xdf, onehot_cols, trimmed_ccsrs):
#   # add a column with only the target set of CCSRs
#   if 'CCSRS' in onehot_cols:
#     Xdf['Trimmed_CCSRS'] = Xdf['CCSRS'].apply(lambda row: [cc for cc in row if cc in trimmed_ccsrs])
#     onehot_cols = list(map(lambda item: item.replace('CCSRS', 'Trimmed_CCSRS'), onehot_cols))
#   # add a column with only the ICD10s of the target CCSR set
#   if 'ICD10S' in onehot_cols:
#     Xdf['Trimmed_ICD10S'] = Xdf[['CCSRS', 'ICD10S']].apply(
#       lambda row: [row['ICD10S'][i] for i in range(len(row['ICD10S'])) if row['CCSRS'][i] in trimmed_ccsrs], axis=1)
#     onehot_cols = list(map(lambda item: item.replace('ICD10S', 'Trimmed_ICD10S'), onehot_cols))
#   return Xdf, onehot_cols
#
#
# # Apply one-hot encoding to the designated columns
# def onehot_encode_cols(Xdf, onehot_cols, onehot_dtypes):
#   for oh_col, dtype in zip(onehot_cols, onehot_dtypes):
#     oh_prefix = oh_col if oh_col not in globals.DRUG_COLS else 'MED%s_' % list(filter(str.isdigit, oh_col))[0]
#     if dtype == str:
#       dummies = pd.get_dummies(Xdf[oh_col], prefix=oh_prefix)
#     elif dtype == list:  # Expand list to (row_id, oh_col indicator) first
#       s = Xdf[oh_col].explode()
#       dummies = pd.crosstab(s.index, s).add_prefix(oh_prefix[:-1] + '_')
#       dummies[dummies > 1] = 1  # in case a list contains duplicates  TODO: double check
#     else:
#       raise NotImplementedError("Cannot encode column '%s' with a data type of '%s'" % (oh_col, dtype))
#     Xdf = Xdf.drop(columns=[oh_col]).join(dummies).fillna(0)
#   return Xdf
#
#
# # Note: column order should be taken care of after calling this function
# def match_Xdf_cols_to_target_features(Xdf, target_features):
#   Xdf_cols = Xdf.columns.to_list()
#
#   # Drop rows that has certain indicator columns not covered in the target feature list (e.g. an unseen CPT code)
#   new_ftrs = set(Xdf_cols) - set(target_features) - set(globals.NON_NUMERIC_COLS)
#   # TODO: think about how to handle different types of unseen codes (e.g. always drop if pproc is unseen, but unseen CCSR/CPT could be fine)
#   if len(new_ftrs) > 0:
#     case_idxs_with_new_ftrs = set()
#     for new_ftr in new_ftrs:
#       idxs = Xdf.index[Xdf[new_ftr] == 1].to_list()
#       case_idxs_with_new_ftrs = case_idxs_with_new_ftrs.union(set(idxs))
#     print_feature_match_details(new_ftrs, 'unseen')
#     print("Dropping %d cases with new features..." % len(case_idxs_with_new_ftrs))
#     Xdf = Xdf.drop(index=list(case_idxs_with_new_ftrs)) \
#       .drop(columns=list(new_ftrs)) \
#       .reset_index(drop=True)
#     if Xdf.shape[0] == 0:
#       raise Exception("All cases in this dataset contain at least 1 unseen indicator!")
#
#   # Add unobserved indicators as columns of 0
#   uncovered_ftrs = set(target_features) - set(Xdf_cols)
#   Xdf[list(uncovered_ftrs)] = 0.0
#   print_feature_match_details(uncovered_ftrs, 'uncovered')
#   return Xdf
#
#
# def print_feature_match_details(feature_set, ftr_type='unseen'):
#   print("\nTotal %s features: %d" % (ftr_type, len(feature_set)))
#   print("#%s CPTs: %d" % (ftr_type, len(list(filter(lambda x: x.startswith('CPT'), feature_set)))))
#   print("#%s CPT Groups: %d" % (ftr_type, len(list(filter(lambda x: x.startswith('CPT_GROUP'), feature_set)))))
#   print("#%s CCSRs: %d" % (ftr_type, len(list(filter(lambda x: x.startswith('CCSR'), feature_set)))))
#
#
# def discretize_columns_df(Xdf: pd.DataFrame, discretize_cols, inplace=False):
#   if not inplace:
#     Xdf = Xdf.copy()
#
#   # Modify data matrix with discretized columns by request
#   for dis_col in discretize_cols:
#     if dis_col not in Xdf.columns.to_list():
#       raise Warning("%s is not in Xdf columns!" % dis_col)
#     elif dis_col == 'AGE_AT_PROC_YRS':
#       Xdf[dis_col] = pd.cut(Xdf[dis_col], bins=globals.AGE_BINS, labels=False, right=False, include_lowest=True)
#     elif dis_col == 'WEIGHT_ZSCORE':
#       weightz_bins = [float('-inf'), -4, -2, -1, 1, 2, 4, float('inf')]
#       print("Weight z-score bins: ", weightz_bins)
#       Xdf[dis_col] = pd.cut(Xdf[dis_col], bins=weightz_bins, labels=False, right=False, include_lowest=True)
#     else:
#       raise Warning("%s discretization is not available yet!" % dis_col)
#   return Xdf
#
#
# def discretize_columns(X, feature_names, discretize_cols, inplace=False):
#   if not inplace:
#     X = np.copy(X)
#
#   # Modify data matrix with discretized columns by request
#   for dis_col in discretize_cols:
#     idx = feature_names.index(dis_col)
#     if dis_col == 'AGE_AT_PROC_YRS':
#       X[:, idx] = np.digitize(X[:, idx], globals.AGE_BINS)
#     elif dis_col == 'WEIGHT_ZSCORE':
#       weightz_bins = [float('-inf'), -4, -2, -1, 1, 2, 4, float('inf')]
#       print("Weight z-score bins: ", weightz_bins)
#       X[:, idx] = np.digitize(X[:, idx], weightz_bins)
#     else:
#       raise Warning("%s discretization is not available yet!" % dis_col)
#   return X
#



# # TODO: add data cleaning, e.g. 1. check for negative LoS and remove
# # Generate data matrix X and a response vector y
# def gen_Xy(df, outcome, cols=globals.FEATURE_COLS, onehot_cols=None, onehot_dtypes=None, trimmed_ccsr=None,
#            discretize_cols=None, is_train=True, target_features=None, remove_o2m=True):
#   """
#   Generate X, y for downstream modeling
#
#   :param df:
#   :param outcome:
#   :param nranges: A list of non-negative numbers, starting with 0
#   :param cols:
#   :return:
#   """
#   # Preprocess outcome values / SPS prediction in df
#   df = preprocess_y(df, outcome)
#   if globals.SPS_LOS_FTR in df.columns.to_list():
#     df = preprocess_y(df, outcome, sps_y=True)
#
#   # Generate a preprocessed data matrix X, and a response vector y
#   o2m_df, med_decile = None, None
#   if is_train:
#     X, features, X_case_keys, y, o2m_df = preprocess_Xtrain(df, outcome, self.FeatureEngineer, cols, remove_o2m=remove_o2m)
#   else:
#     X, features, X_case_keys = preprocess_Xtest(df, target_features, cols, onehot_cols, onehot_dtypes, trimmed_ccsr,
#                                                 discretize_cols, remove_o2m)
#     y = df[df['SURG_CASE_KEY'].isin(X_case_keys)][outcome].to_numpy()
#
#   return X, y, features, X_case_keys, o2m_df, med_decile
#