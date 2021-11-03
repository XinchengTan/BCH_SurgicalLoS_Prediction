"""Helper functions to preprocess the data and generate data matrix with its corresponding labels"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import globals
from . import data_prepare as dp


class Dataset(object):

  def __init__(self, df, outcome=globals.LOS, cols=globals.FEATURE_COLS, onehot_cols=None, onehot_dtypes=None,
               test_pct=0.2):
    assert len(onehot_cols) == len(onehot_dtypes), "One-hot Encoding columns and dtypes must match!"
    X, y, feature_cols, case_keys = gen_Xy(df, outcome, cols, onehot_cols, onehot_dtypes)
    # nan_cases = set(np.argwhere(np.isnan(X))[:, 0])
    # print("Number of cases with NaN in data matrix:", len(nan_cases))
    # print(nan_cases)
    self.Xtrain, self.Xtest, self.ytrain, self.ytest, self.train_idx, self.test_idx = gen_train_test(X, y, test_pct)
    self.feature_names = feature_cols
    self.case_keys = case_keys
    self.sps_preds = df[globals.SPS_LOS_FTR].to_numpy()  # Might contain NaN

  def __str__(self):
    res = "Training set size: %d\n" \
          "Test set size: %d\n"\
          "Number of features: %d\n\n" % (self.ytrain.shape[0], self.ytest.shape[0], len(self.feature_names))
    res += "Feature names: \n" + "\n".join(self.feature_names)
    return res



# TODO: add data cleaning, e.g. check for negative LoS and remove it
# Generate data matrix X
def gen_Xy(df, outcome=globals.LOS, cols=globals.FEATURE_COLS, onehot_cols=None, onehot_dtypes=None):
  """
  Generate X, y for downstream modeling

  :param df:
  :param outcome:
  :param nranges: A list of non-negative numbers, starting with 0
  :param cols:
  :return:
  """
  # Make data matrix X
  X = df[cols]
  X.loc[(X.SEX_CODE != 'F'), 'SEX_CODE'] = 0.0
  X.loc[(X.SEX_CODE == 'F'), 'SEX_CODE'] = 1.0
  feature_cols = list(cols)
  if onehot_cols is not None:
    # Apply one-hot encoding to the designated columns
    for oh_col, dtype in zip(onehot_cols, onehot_dtypes):
      if dtype == str:  # can directly use get_dummies()
        dummies = pd.get_dummies(X[oh_col], prefix=oh_col)
      elif dtype == list:  # Need to expand list to (row_id, oh_col indicator) first TODO: Bug somewhere leading to a lot of NaNs!!
        # dummies = pd.get_dummies(
        #   pd.DataFrame(X[oh_col].to_list()).stack(), prefix=oh_col
        # ).astype(int).sum(level=0)
        s = X[oh_col].explode()
        dummies = pd.crosstab(s.index, s)
      else:
        raise NotImplementedError
      X = X.drop(columns=[oh_col]).join(dummies)
      feature_cols.remove(oh_col)
      feature_cols.extend(dummies.columns.to_list())
  #dp.print_df_info(X, 'Data Matrix (pd DF)')

  # Drop SURG_CASE_KEY
  X_case_key = X['SURG_CASE_KEY'].to_numpy()
  X = X.drop(columns=['SURG_CASE_KEY'])
  feature_cols.remove('SURG_CASE_KEY')

  # Bucket SPS predicted LoS into 9 classes, if there is such column
  if globals.SPS_LOS_FTR in cols:  # Assume SPS prediction are all integers?
    X.loc[X[globals.SPS_LOS_FTR] > globals.MAX_NNT] = globals.MAX_NNT + 1
  X = X.to_numpy(dtype=np.float64)

  # Make response vector y
  y = np.array(df.LENGTH_OF_STAY.to_numpy())
  if outcome == globals.LOS:
    return X, y
  elif outcome == ">12h":
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
  elif outcome == ">1d":
    y[y > 1] = 1
    y[y <= 1] = 0
  elif outcome == "NNT":
    y = gen_y_nnt(df['NUM_OF_NIGHTS'])
  elif outcome.endswith("nnt"):
    cutoff = int(outcome.split("nnt")[0])
    y = gen_y_nnt_binary(y, cutoff)
  return X, y, feature_cols, X_case_key


def gen_y_nnt(dfcol):
  y = np.array(dfcol.to_numpy())
  y[y > globals.MAX_NNT] = globals.MAX_NNT + 1
  return y


def gen_y_nnt_binary(y, cutoff):
  yb = np.copy(y)
  yb[yb < cutoff] = 0
  yb[yb >= cutoff] = 1
  return yb


# Perform train-test split
def gen_train_test(X, y, test_pct=0.2):
  X, y = pd.DataFrame(X), pd.Series(y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, random_state=globals.SEED)
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

