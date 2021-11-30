"""
Helper functions to preprocess the data and generate data matrix with its corresponding labels
"""

from IPython.display import display
from imblearn.over_sampling import SMOTENC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import globals
from . import data_prepare as dp


class Dataset(object):

  def __init__(self, df, outcome=globals.NNT, cols=globals.FEATURE_COLS, onehot_cols=None, onehot_dtypes=None,
               test_pct=0.2, cohort=globals.COHORT_ALL, trimmed_ccsr=None):
    if (onehot_cols is not None) and (onehot_dtypes is not None):
      assert len(onehot_cols) == len(onehot_dtypes), "One-hot Encoding columns and dtypes must match!"
    self.df = df
    self.cohort = cohort
    if cohort != globals.COHORT_ALL:  # Filter df by primary procedure cohort
      ch_pprocs = globals.COHORT_TO_PPROCS[cohort]
      self.cohort_df = df.query("PRIMARY_PROC in @ch_pprocs")
    else:
      self.cohort_df = df

    X, y, feature_cols, case_keys = gen_Xy(self.cohort_df, outcome, cols, onehot_cols, onehot_dtypes, trimmed_ccsr)
    self.Xtrain, self.Xtest, self.ytrain, self.ytest, self.train_idx, self.test_idx = gen_train_test(X, y, test_pct)
    self.feature_names = feature_cols
    self.case_keys = case_keys
    self.sps_preds = df[globals.SPS_LOS_FTR].to_numpy()  # Might contain NaN
    self.cpt_groups = df['CPT_GROUPS'].to_numpy()

  def __str__(self):
    res = "Training set size: %d\n" \
          "Test set size: %d\n"\
          "Number of features: %d\n\n" % (self.ytrain.shape[0], self.ytest.shape[0], len(self.feature_names))
    res += "Feature names: \n" + "\n".join(self.feature_names)
    return res


# TODO: add data cleaning, e.g. check for negative LoS and remove it
# Generate data matrix X
def gen_Xy(df, outcome=globals.NNT, cols=globals.FEATURE_COLS, onehot_cols=None, onehot_dtypes=None, trimmed_ccsr=None):
  """
  Generate X, y for downstream modeling

  :param df:
  :param outcome:
  :param nranges: A list of non-negative numbers, starting with 0
  :param cols:
  :return:
  """
  # Make data matrix X
  X, feature_cols, X_case_key = gen_X(df, cols, onehot_cols, onehot_dtypes, trimmed_ccsr=trimmed_ccsr)
  X = X.to_numpy(dtype=np.float64)

  # Get outcome vector
  y = gen_y(df, outcome)
  return X, y, feature_cols, X_case_key


def gen_X(df, cols=globals.FEATURE_COLS, onehot_cols=None, onehot_dtypes=None, remove_nonnumeric=True, verbose=False, trimmed_ccsr=None):
  # Make data matrix X numeric
  X = df[cols]
  X.loc[(X.SEX_CODE != 'F'), 'SEX_CODE'] = 0.0
  X.loc[(X.SEX_CODE == 'F'), 'SEX_CODE'] = 1.0

  feature_cols = list(cols)
  if trimmed_ccsr and 'CCSRS' in onehot_cols:  # add a column with only the target set of CCSRs
    #trimmed_ccsr_wPrefix = {'CCSR_' + c for c in trimmed_ccsr}
    X['Trimmed_CCSRS'] = X['CCSRS'].apply(lambda row: [cc for cc in row if cc in trimmed_ccsr])
    onehot_cols = [col if col != 'CCSRS' else 'Trimmed_CCSRS' for col in onehot_cols]
    feature_cols = [col if col != 'CCSRS' else 'Trimmed_CCSRS' for col in feature_cols]
    # add a column with only the ICD10 of the target CCSR set
    if 'ICD10S' in onehot_cols:
      X['Trimmed_ICD10S'] = X[['CCSRS', 'ICD10S']].apply(lambda row: [row['ICD10S'][i]
                                                                      for i in range(len(row['ICD10S']))
                                                                      if row['CCSRS'][i] in trimmed_ccsr], axis=1)
      onehot_cols = [col if col != 'ICD10S' else 'Trimmed_ICD10S' for col in onehot_cols]
      feature_cols = [col if col != 'ICD10S' else 'Trimmed_ICD10S' for col in feature_cols]

  if onehot_cols is not None:
    # Apply one-hot encoding to the designated columns
    for oh_col, dtype in zip(onehot_cols, onehot_dtypes):
      if dtype == str:  # can directly use get_dummies()
        dummies = pd.get_dummies(X[oh_col], prefix=oh_col)
      elif dtype == list:  # Need to expand list to (row_id, oh_col indicator) first
        s = X[oh_col].explode()
        dummies = pd.crosstab(s.index, s).add_prefix(oh_col[:-1] + '_')
        dummies[dummies > 1] = 1  # in case a list contains duplicates
        # # Alternative:
        # dummies = X[oh_col].apply(lambda x: pd.Series(1, x))
        # X = pd.concat([X.drop(columns=[oh_col]), dummies.fillna(0)], axis=1)
      else:
        raise NotImplementedError
      X = X.drop(columns=[oh_col]).join(dummies).fillna(0)
      feature_cols.remove(oh_col)
      feature_cols.extend(dummies.columns.to_list())

  # Save SURG_CASE_KEY, but drop with other non-numeric columns for data matrix
  X_case_key = X['SURG_CASE_KEY'].to_numpy()
  if remove_nonnumeric:
    X.drop(columns=globals.NON_NUMERIC_COLS, inplace=True, errors='ignore')  #
    for nnm_col in globals.NON_NUMERIC_COLS:
      if nnm_col in feature_cols:
        feature_cols.remove(nnm_col)

  # Bucket SPS predicted LoS into 9 classes, if there such prediction exists
  if globals.SPS_LOS_FTR in cols:  # Assume SPS prediction are all integers?
    X.loc[(X[globals.SPS_LOS_FTR] > globals.MAX_NNT), globals.SPS_LOS_FTR] = globals.MAX_NNT + 1
  if verbose:
    display(X.head(20))
  assert X.shape[1] == len(feature_cols), 'Generated data matrix has %d features, but feature list has %d items' % \
                                          (X.shape[1], len(feature_cols))
  return X, feature_cols, X_case_key


def gen_y(df, outcome):
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
  return y


def gen_y_nnt(dfcol):
  y = np.array(dfcol.to_numpy())
  y[y > globals.MAX_NNT] = globals.MAX_NNT + 1
  return y


def gen_y_nnt_binary(y, cutoff):  # task: predict if LoS <= cutoff (cutoff in range(0, 8)
  yb = np.copy(y)
  yb[y <= cutoff] = 1
  yb[y > cutoff] = 0
  return yb


# Perform train-test split
def gen_train_test(X, y, test_pct=0.2):
  """
  Returns the purely integer-location based index w.r.t. the data matrix X for the train and test set respectively.
  i.e. max(X_train.index) <= n_samples - 1 and max(X_test.index) <= n_samples - 1
  These index lists are not the row index label of the original df, because X is already a numeric matrix and
  has lost the index labels during the conversion.

  :param X: a numeric matrix (n_samples, n_features)
  :param y: a response vector (n_samples, )
  :param test_pct: desired test set percentage, a float between 0 and 1

  :return: training and test set data matrix, response vector and location-based index w.r.t. X
  """
  X, y = pd.DataFrame(X), pd.Series(y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, random_state=globals.SEED)
  # print("[gen_train_test] X index: ", X.index)
  # print("min index, max index: ", min(X.index), max(X.index))
  # print("train - min, max: ", min(X_train.index), max(X_train.index))
  # print("test - min, max: ", min(X_test.index), max(X_test.index))
  return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), X_train.index, X_test.index


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
