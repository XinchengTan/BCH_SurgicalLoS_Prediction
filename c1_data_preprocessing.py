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



class Dataset(object):

  def __init__(self, df, outcome=globals.NNT, cols=globals.FEATURE_COLS, onehot_cols=None, onehot_dtypes=None,
               test_pct=0.2, test_idxs=None, cohort=globals.COHORT_ALL, trimmed_ccsr=None, discretize_cols=None,
               basic_Xy=None, remove_o2m=(True, True)):
    if (onehot_cols is not None) and (onehot_dtypes is not None):
      assert len(onehot_cols) == len(onehot_dtypes), "One-hot Encoding columns and dtypes must match!"
    self.df = df
    self.cohort = cohort
    if cohort != globals.COHORT_ALL:  # Filter df by primary procedure cohort
      ch_pprocs = globals.COHORT_TO_PPROCS[cohort]
      self.cohort_df = df.query("PRIMARY_PROC in @ch_pprocs")
    else:
      self.cohort_df = df

    # 1. Train-test split
    if test_idxs is None:
      train_df, test_df = self.cohort_df, None if test_pct == 0 else train_test_split(self.cohort_df, test_size=test_pct)
    else:
      test_df = self.cohort_df.iloc[test_idxs]
      train_df = self.cohort_df.iloc[list(set(range(self.cohort_df.shape[0])) - set(test_idxs))]

    # 2. Preprocess training set
    self.Xtrain, self.ytrain, self.feature_names, self.train_case_keys, self.o2m_df_train = gen_Xy(
      train_df, outcome, cols, onehot_cols, onehot_dtypes, trimmed_ccsr, discretize_cols, True, remove_o2m[0])
    self.train_cohort_df = self.cohort_df[self.cohort_df['SURG_CASE_KEY'].isin(self.train_case_keys)]

    # 3. Preprocess test set, if it's not empty
    if test_df is not None:
      self.Xtest, self.ytest, _, self.test_case_keys, _ = gen_Xy(
        test_df, outcome, cols, onehot_cols, onehot_dtypes, trimmed_ccsr, discretize_cols, False, self.feature_names,
        self.o2m_df_train if remove_o2m[1] else None)
      self.test_cohort_df = self.cohort_df[self.cohort_df['SURG_CASE_KEY'].isin(self.test_case_keys)]
    else:
      self.Xtest, self.ytest, self.test_case_keys = np.array([]), np.array([]), np.array([])
      self.test_cohort_df = test_df

    # self.case_keys = case_keys
    # self.sps_preds = df[globals.SPS_LOS_FTR].to_numpy()  # Might contain NaN
    # self.cpt_groups = df['CPT_GROUPS'].to_numpy()

  def _filter_X_by_index(self, Xtype, keep_idxs):
    if Xtype == globals.XTEST:
      self.Xtest = self.Xtest[keep_idxs, :]
      self.ytest = self.ytest[keep_idxs]
      self.test_idx = self.test_idx[keep_idxs]
    else:
      raise NotImplementedError

  def __str__(self):
    res = "Training set size: %d\n" \
          "Test set size: %d\n"\
          "Number of features: %d\n\n" % (self.ytrain.shape[0], self.ytest.shape[0], len(self.feature_names))
    res += "Feature names: \n" + "\n".join(self.feature_names)
    return res


# TODO: add data cleaning, e.g. 1. check for negative LoS and remove
# Generate data matrix X and a response vector y
def gen_Xy(df, outcome, cols=globals.FEATURE_COLS, onehot_cols=None, onehot_dtypes=None, trimmed_ccsr=None,
           discretize_cols=None, is_train=True, target_features=None, remove_o2m=True):
  """
  Generate X, y for downstream modeling

  :param df:
  :param outcome:
  :param nranges: A list of non-negative numbers, starting with 0
  :param cols:
  :return:
  """
  # Preprocess outcome values in df
  df = preprocess_y(df, outcome)

  # Generate a preprocessed data matrix X, and a response vector y
  o2m_df = None
  if is_train:
    X, features, X_case_keys, y, o2m_df = preprocess_Xtrain(
      df, outcome, cols, onehot_cols, onehot_dtypes, trimmed_ccsr=trimmed_ccsr, discretize_cols=discretize_cols,
      remove_o2m=remove_o2m)
  else:
    X, features, X_case_keys = preprocess_Xtest(df, remove_o2m, target_features, cols, onehot_cols, onehot_dtypes, discretize_cols)
    y = df[df['SURG_CASE_KEY'].isin(X_case_keys)][outcome].to_numpy()

  return X, y, features, X_case_keys, o2m_df


def preprocess_Xtrain(df, outcome, cols=globals.FEATURE_COLS, onehot_cols=None, onehot_dtypes=None,
                      remove_nonnumeric=True, verbose=False, trimmed_ccsr=None, discretize_cols=None, remove_o2m=True):
  # Make data matrix X numeric
  X = df.copy()[cols]
  X.loc[(X.SEX_CODE != 'F'), 'SEX_CODE'] = 0.0
  X.loc[(X.SEX_CODE == 'F'), 'SEX_CODE'] = 1.0

  if trimmed_ccsr:
    X, onehot_cols = trim_ccsr_in_X(X, onehot_cols, trimmed_ccsr)

  if onehot_cols is not None:
    X = onehot_encode_cols(X, onehot_cols, onehot_dtypes)

  # Save SURG_CASE_KEY, but drop with other non-numeric columns for data matrix
  X_case_keys = X['SURG_CASE_KEY'].to_numpy()
  if remove_nonnumeric:
    X.drop(columns=globals.NON_NUMERIC_COLS, inplace=True, errors='ignore')

  # Bucket SPS predicted LoS into 9 classes, if there such prediction exists
  if globals.SPS_LOS_FTR in cols:  # Assume SPS prediction are all integers
    X.loc[(X[globals.SPS_LOS_FTR] > globals.MAX_NNT), globals.SPS_LOS_FTR] = globals.MAX_NNT + 1

  target_features = X.columns.to_list()

  # Discretize certain continuous columns by request
  X = X.to_numpy(dtype=np.float64)
  if discretize_cols:
    discretize_columns(X, target_features, discretize_cols, inplace=True)

  # Remove cases that have multiple possible outcomes
  o2m_df, y = None, df[outcome].to_numpy()
  if remove_o2m == True:
    s = time()
    o2m_df, X, X_case_keys, y = discard_o2m_cases_from_self(X, X_case_keys, y, target_features)  # training set
    print("Removing o2m cases from self took %d sec" % (time() - s))

  if verbose:
    display(pd.DataFrame(X, columns=target_features).head(20))
  assert X.shape[1] == len(target_features), 'Generated data matrix has %d features, but feature list has %d items' % \
                                          (X.shape[1], len(target_features))  # Basic sanity check
  return X, target_features, X_case_keys, y, o2m_df


# Data preprocessing (clean, discretize and one-hot encode certain features)
def preprocess_Xtest(df, skip_cases_df_or_fp, feature_cols, target_features, onehot_cols=None, onehot_dtypes=None,
                     trimmed_ccsr=None, discretize_cols=None):
  # Get target feature names from training data
  if skip_cases_df_or_fp is None:
    skip_cases_df = None
  elif isinstance(skip_cases_df_or_fp, pd.DataFrame):
    skip_cases_df = skip_cases_df_or_fp
  else:
    skip_cases_df = pd.read_csv(skip_cases_df_or_fp)

  # Make data matrix X numeric
  X = df.copy()[feature_cols]
  X.loc[(X.SEX_CODE != 'F'), 'SEX_CODE'] = 0.0
  X.loc[(X.SEX_CODE == 'F'), 'SEX_CODE'] = 1.0

  # Add a column of trimmed CCSRs with/without a column of the corresponding trimmed ICD10s
  if trimmed_ccsr:
    X, onehot_cols = trim_ccsr_in_X(X, onehot_cols, trimmed_ccsr)

  if onehot_cols is not None:
    # One-hot encode the required columns according to a given historical set of features (e.g. CPT, CCSR etc.)
    X = onehot_encode_cols(X, onehot_cols, onehot_dtypes)

    # Match Xdf to the target features from the training data matrix (for now, only medical codes lead to such need)
    X = match_Xdf_cols_to_target_features(X, target_features)

  # Save SURG_CASE_KEY, but drop with other non-numeric columns for data matrix
  X_case_key = X['SURG_CASE_KEY'].to_numpy()
  X.drop(columns=globals.NON_NUMERIC_COLS, inplace=True, errors='ignore')

  # Reorder columns such that it aligns with target_features
  X = X[target_features]

  # Discretize certain continuous columns by request
  X = X.to_numpy(dtype=np.float64)
  if discretize_cols:
    discretize_columns(X, target_features, discretize_cols, inplace=True)

  # Remove cases identical to any in skip_cases, update X_case_keys correspondingly
  if skip_cases_df:
    s = time()
    X, X_case_key = discard_o2m_cases_from_historical_data(X, X_case_key, skip_cases_df)
    print("Removing o2m cases from training set took %d sec" % (time() - s))

  # Basic sanity check
  assert X.shape[1] == len(target_features), 'Generated data matrix has %d features, but feature list has %d items' % \
                                             (X.shape[1], len(target_features))
  return X, target_features, X_case_key


def trim_ccsr_in_X(Xdf, onehot_cols, trimmed_ccsrs):
  # add a column with only the target set of CCSRs
  if 'CCSRS' in onehot_cols:
    Xdf['Trimmed_CCSRS'] = Xdf['CCSRS'].apply(lambda row: [cc for cc in row if cc in trimmed_ccsrs])
    onehot_cols = list(map(lambda item: item.replace('CCSRS', 'Trimmed_CCSRS'), onehot_cols))
  # add a column with only the ICD10s of the target CCSR set
  if 'ICD10S' in onehot_cols:
    Xdf['Trimmed_ICD10S'] = Xdf[['CCSRS', 'ICD10S']].apply(
      lambda row: [row['ICD10S'][i] for i in range(len(row['ICD10S'])) if row['CCSRS'][i] in trimmed_ccsrs], axis=1)
    onehot_cols = list(map(lambda item: item.replace('ICD10S', 'Trimmed_ICD10S'), onehot_cols))
  return Xdf, onehot_cols


# Apply one-hot encoding to the designated columns
def onehot_encode_cols(Xdf, onehot_cols, onehot_dtypes):
  for oh_col, dtype in zip(onehot_cols, onehot_dtypes):
    if dtype == str:
      dummies = pd.get_dummies(Xdf[oh_col], prefix=oh_col)
    elif dtype == list:  # Expand list to (row_id, oh_col indicator) first
      s = Xdf[oh_col].explode()
      dummies = pd.crosstab(s.index, s).add_prefix(oh_col[:-1] + '_')
      dummies[dummies > 1] = 1  # in case a list contains duplicates  TODO: double check
    else:
      raise NotImplementedError("Cannot encode column '%s' with a data type of '%s'" % (oh_col, dtype))
    Xdf = Xdf.drop(columns=[oh_col]).join(dummies).fillna(0)
  return Xdf


# Note: column order should be taken care of after calling this function
def match_Xdf_cols_to_target_features(Xdf, target_features):
  Xdf_cols = Xdf.columns.to_list()

  # Drop rows that has certain indicator columns not covered in the target feature list (e.g. an unseen CPT code)
  new_ftrs = set(Xdf_cols) - set(target_features) - set(globals.NON_NUMERIC_COLS)
  # TODO: think about how to handle different types of unseen codes (e.g. always drop if pproc is unseen, but unseen CCSR/CPT could be fine)
  if len(new_ftrs) > 0:
    case_idxs_with_new_ftrs = set()
    for new_ftr in new_ftrs:
      idxs = Xdf.index[Xdf[new_ftr] == 1].to_list()
      case_idxs_with_new_ftrs = case_idxs_with_new_ftrs.union(set(idxs))
    print_feature_match_details(new_ftrs, 'unseen')
    print("Dropping %d cases with new features..." % len(case_idxs_with_new_ftrs))
    Xdf = Xdf.drop(index=list(case_idxs_with_new_ftrs)) \
      .drop(columns=list(new_ftrs)) \
      .reset_index(drop=True)
    if Xdf.shape[0] == 0:
      raise Exception("All cases in this dataset contain at least 1 unseen indicator!")

  # Add unobserved indicators as columns of 0
  uncovered_ftrs = set(target_features) - set(Xdf_cols)
  Xdf[list(uncovered_ftrs)] = 0.0
  print_feature_match_details(uncovered_ftrs, 'uncovered')
  return Xdf


def print_feature_match_details(feature_set, ftr_type='unseen'):
  print("\nTotal %s features: %d" % (ftr_type, len(feature_set)))
  print("#%s CPTs: %d" % (ftr_type, len(list(filter(lambda x: x.startswith('CPT'), feature_set)))))
  print("#%s CPT Groups: %d" % (ftr_type, len(list(filter(lambda x: x.startswith('CPT_GROUP'), feature_set)))))
  print("#%s CCSRs: %d" % (ftr_type, len(list(filter(lambda x: x.startswith('CCSR'), feature_set)))))


def discretize_columns(X, feature_names, discretize_cols, inplace=False):
  if not inplace:
    X = np.copy(X)

  # Modify data matrix with discretized columns by request
  for dis_col in discretize_cols:
    idx = feature_names.index(dis_col)
    if dis_col == 'AGE_AT_PROC_YRS':
      X[:, idx] = np.digitize(X[:, idx], globals.AGE_BINS)
    elif dis_col == 'WEIGHT_ZSCORE':
      weightz_bins = [float('-inf'), -4, -2, -1, 1, 2, 4, float('inf')]
      print("Weight z-score bins: ", weightz_bins)
      X[:, idx] = np.digitize(X[:, idx], weightz_bins)
    else:
      raise Warning("%s discretization is not available yet!" % dis_col)
  return X


def preprocess_y(df, outcome):
  # Generate an outcome vector y with shape: (n_samples, )
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
  o2m_df = Xydf_dup\
    .groupby(by=features)\
    .filter(lambda x: len(x.value_counts().index) > 1)
  print("Covered %d o2m cases" % o2m_df.shape[0])
  if unique:
    o2m_df = o2m_df.drop_duplicates(subset=features)
    print("Contains %d unique o2m cases" % o2m_df.shape[0])
  o2m_df.drop(columns=['Outcome'], inplace=True)

  if save_fp:
    o2m_df.to_csv(save_fp, index=False)
  return o2m_df


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

  return Xdf[skip_cases_df.columns].to_numpy(), Xdf['SURG_CASE_KEY'].to_numpy()


# Removes cases that has multiple possible outcomes from input data
def discard_o2m_cases_from_self(X, X_case_keys, y, features):
  o2m_df = gen_o2m_cases(X, y, features, unique=False)
  o2m_idx = o2m_df.index.to_list()
  #print("One-to-many cases index: ", o2m_idx)
  print("#O2M cases: ", len(o2m_idx))
  idxs = np.delete(np.arange(X.shape[0]), o2m_idx)
  X, X_case_keys, y = X[idxs, :], X_case_keys[idxs], y[idxs]
  return o2m_df.drop_duplicates(subset=features), X, X_case_keys, y




  # # Generate o2m cases from training set
  # if skip_o2m_from_train:
  #   skip_cases_df, X_train, y_train, train_idx = discard_o2m_cases_from_self(X_train, y_train, features)
  #   X_test, test_idx = discard_o2m_cases_from_historical_data(X_test.to_numpy(), test_idx, skip_cases_df)
  #   # TODO: drop unseen cols, add unobserved cols (train-test split before gen_feature_matrix???)
  #   y_test = y_test[test_idx]
  #   X_case_keys = ???


 # if test_idxs is not None:
    #   self.test_idx, self.train_idx = np.array(test_idxs), np.array(list(set(range(df.shape[0])) - set(test_idxs)))
    #   self.Xtrain, self.Xtest, self.ytrain, self.ytest = X[self.train_idx, :], X[self.test_idx, :], y[self.train_idx], y[self.test_idx]
    # else:
    #   if test_pct > 0:
    #     self.Xtrain, self.Xtest, self.ytrain, self.ytest, self.train_idx, self.test_idx = gen_train_test(X, y, test_pct)
    #   else:
    #     self.Xtrain, self.ytrain, self.train_idx = X, y, np.arange(X.shape[0])
    #     self.Xtest, self.ytest, self.test_idx = np.array([]), np.array([]), np.array([])


# def gen_o2m_cases(X, y, features, drop_duplicates=False):
#   Xydf = pd.DataFrame(X, columns=features)
#   Xydf['Outcome'] = y
#   dup_mask = Xydf.duplicated(subset=features, keep=False)
#   Xydf_dup = Xydf[dup_mask]
#
#   # Get cases with identical features but different outcomes
#   o2m_df = Xydf_dup.groupby(by=features)\
#     .filter(lambda x: len(x.value_counts().index) > 1)
#
#   if drop_duplicates:
#     return o2m_df.drop_duplicates(subset=features)
#   return o2m_df




# def denoise(X, y, features, how=globals.DENOISE_DEL_O2M):
#   """
#   Denoise dataset (X, y) by removing pure duplicates,
#   or coalescing one-to-many cases to one case with its majority outcome,
#   or doing both operations.
#
#   :param X:
#   :param y:
#   :param noisy_cases: If not None, denoise these cases from X
#   :param how:
#   :return: A dataframe of X, y, with the index of the original X preserved
#   """
#   # Make Xydf_dup and Xydf_nodup
#   Xydf = pd.DataFrame(X, columns=features)
#   Xydf['Outcome'] = y
#   dup_mask = Xydf.duplicated(subset=features, keep=False)
#   Xydf_dup = Xydf[dup_mask]
#
#   Xydf_clean = Xydf.drop_duplicates(subset=features, keep=False)
#
#   o2m_keep_df = remove_o2m_cases(Xydf_dup, features)
#   pure_dup_keep_df = remove_pure_dups(Xydf_dup, features)
#
#   if how == globals.DENOISE_ALL:
#     cleaned_cases_df = pd.concat([o2m_keep_df, pure_dup_keep_df])
#     Xydf_clean = pd.concat([Xydf_clean, cleaned_cases_df])
#   elif how == globals.DENOISE_O2M:
#     cleaned_cases_df = o2m_keep_df
#     #print("\n pure dup df: ", pure_dup_keep_df.columns)
#     #print("\n Xydf: ", Xydf_dup.columns)
#     kept_noise_df = pd.merge(Xydf_dup, pure_dup_keep_df, on=features, how='left', indicator=True, suffixes=('', '_r'))\
#       .loc[lambda x: x['_merge'] != 'both']
#     #print("\n kept noise df: ", kept_noise_df.columns)
#     Xydf_clean = pd.concat([Xydf_clean, cleaned_cases_df, kept_noise_df[features + ['Outcome']]])
#   elif how == globals.DENOISE_PURE_DUP:
#     cleaned_cases_df = pure_dup_keep_df
#     kept_noise_df = pd.merge(Xydf_dup, o2m_keep_df, on=features, how='left', indicator=True, suffixes=('', '_r'))\
#       .loc[lambda x: x['_merge'] != 'both']
#     Xydf_clean = pd.concat([Xydf_clean, cleaned_cases_df, kept_noise_df[features + ['Outcome']]])
#
#   else:
#     raise NotImplementedError
#
#   assert len(Xydf_clean.columns) == len(features) + 1, "Clean Xy df has wrong number of columns!"
#   #   assert True not in Xydf_clean.duplicated(subset=features), "Clean Xy df still has cases with duplications!"
#
#   return Xydf_clean, cleaned_cases_df
#
#
# def remove_o2m_cases(Xydup_df, features, noisy_cases_df=None):
#   # o2m_df = Xydup_df.groupby(by=features)\
#   #   .filter(lambda x: len(x.value_counts().index) > 1)
#   #   .groupby(by=features)['Outcome'].count().sort_values('pos').groupby(level=len(features)-1).tail(1)
#   Xydup_df = Xydup_df.groupby(by=features).filter(lambda x: len(x.value_counts().index) > 1)  # o2m cases
#   o2m_grp2idxs = Xydup_df.groupby(by=features).groups
#   o2m_keep_X, y, indices = [], [], []
#   if not isinstance(noisy_cases_df, pd.DataFrame):  # noisy_cases_df == None or its equivalent
#     for ftr, idxs in o2m_grp2idxs.items():
#       outcomes = Xydup_df.loc[idxs]['Outcome'].to_list()
#       outcome_counter = Counter(outcomes)
#       if len(outcome_counter) > 1:
#         o2m_keep_X.append(ftr)
#         cur_y = max(outcome_counter, key=outcome_counter.get)
#         y.append(cur_y)
#         indices.append(idxs[outcomes.index(cur_y)])
#     o2m_keep_df = pd.DataFrame(o2m_keep_X, columns=features, index=indices)
#     o2m_keep_df['Outcome'] = y
#     return o2m_keep_df
#   else:
#     for ftr, idxs in o2m_grp2idxs.items():
#       noisy_case_match = noisy_cases_df[(noisy_cases_df[features] == ftr).all(1)]
#       if len(noisy_case_match) > 0:
#         selected_y = noisy_case_match.iloc[0]['Outcome']
#         ftr_matched_cases = Xydup_df.loc[idxs]
#         outcome_dismatch_cases = ftr_matched_cases[ftr_matched_cases['Outcome'] != selected_y]
#         Xydup_df.drop(index=outcome_dismatch_cases.index.to_list(), inplace=True)
#     Xydup_df.drop_duplicates(keep='first', inplace=True)
#     return Xydup_df
#
#
# def remove_pure_dups(Xydup_df, features, noisy_cases_df=None):
#   if not isinstance(noisy_cases_df, pd.DataFrame):  # noisy_cases_df == None
#     pure_dup_keep_df = Xydup_df.groupby(by=features)\
#       .filter(lambda x: len(x.value_counts().index) == 1)\
#       .drop_duplicates(subset=features, keep='first')
#   else:
#     pure_dup_keep_df = Xydup_df.groupby(by=features) \
#       .filter(lambda x: len(x.value_counts().index) == 1)
#
#
#
#     join_df = pd.merge(pure_dup_keep_df, noisy_cases_df, on=features, how='left', suffixes=('_l', '_r'), indicator=True)
#     pure_dup_keep_df = join_df[join_df['_merge'] == 'left_only'].drop('_merge', axis=1, inplace=True)
#     display(pure_dup_keep_df.head(5))
#     overlap = join_df[join_df['_merge'] == 'both']\
#       .drop_duplicates(subset=features+['Outcome_l'], keep='first')\
#       .rename(columns={'Outcome_l': 'Outcome'}, inplace=True)\
#       .drop(['Outcome_r', '_merge'], axis=1, inplace=True)
#     pure_dup_keep_df = pd.concat([pure_dup_keep_df, overlap])
#
#   return pure_dup_keep_df
#
