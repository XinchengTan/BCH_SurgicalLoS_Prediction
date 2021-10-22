"""Helper functions to preprocess the data and generate data matrix with its corresponding labels"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import globals


# TODO: add data cleaning, e.g. check for negative LoS and remove it
# Generate data matrix X
def gen_Xy(df, outcome=globals.LOS, cols=None):
  """

  :param df:
  :param outcome:
  :param nranges: A list of non-negative numbers, starting with 0
  :param cols:
  :return:
  """
  if cols is None:
    ftr_cols = globals.FEATURE_COLS
  else:
    ftr_cols = cols
  X = df[ftr_cols]
  X.loc[(X.SEX_CODE != 'F'), 'SEX_CODE'] = 0.0
  X.loc[(X.SEX_CODE == 'F'), 'SEX_CODE'] = 1.0
  X = X.to_numpy(dtype=np.float64)

  y = df.LENGTH_OF_STAY.to_numpy()
  if outcome == globals.LOS:
    return X, y
  elif outcome == ">12h":
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
  elif outcome == ">1d":
    y[y > 1] = 1
    y[y <= 1] = 0
  elif outcome == "num_nights":
    y = df['NUM_OF_NIGHTS'].to_numpy()
    y[y > globals.MAX_NNT] = globals.MAX_NNT + 1

  return X, y


# Perform train-test split (treat the test set as validation set)
def gen_train_val_test(X, y, val_pct=0.2, test_pct=0.2):
  X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_pct, random_state=globals.SEED)
  X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_pct, random_state=globals.SEED)
  return X_train, X_val, X_test, y_train, y_val, y_test


def standardize(X_train, X_test=None):
  scaler = StandardScaler().fit(X_train)
  if X_test is None:
    return scaler.transform(X_train)
  return scaler.transform(X_test)

