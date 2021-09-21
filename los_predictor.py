"""
Predictive models for LoS
- regression models
- SVM (start with linear kernel, monitor train/val loss)
- Decision Trees, Random Forest, XGBoost, CART?
- KNN?


Preprocessing:
- normalization
- one-hot encode categorical variables
- mixed-type inputs VS all discrete variables

Input variables:
- demographics info
- diagnosis codes
- commorbidity ?= decile

Output type:
- LoS (continuous)
- range of LoS (discrete & ordinal), e.g. LoS in [0, 1), [1, 2), [2,3), [3, 4) ...
(-- resp, cardio outcome)

Evaluation:
- Confusion matrix
- ROC curve

Analysis:
- feature importance (Shapley value)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor


SEED = 998

# TODO: add data cleaning, e.g. check for negative LoS and remove it

# Generate data matrix X
def gen_Xy(df, outcome="LOS", nranges=None, cols=None):
  """

  :param df:
  :param outcome:
  :param nranges: A list of non-negative numbers, starting with 0
  :param cols:
  :return:
  """
  if cols is None:
    cols = ['SEX_CODE', 'AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE',
            'PROC_DECILE', 'Cardiovascular', 'Digestive',
            'Endocrine', 'Genetic', 'Hematologic', 'Immunologic', 'Infectious',
            'Mental', 'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic',
            'Nutrition', 'Optic', 'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin',
            'Uncategorized', 'Urogenital']
  X = df[cols]
  X.loc[(X.SEX_CODE != 'F'), 'SEX_CODE'] = 0.0
  X.loc[(X.SEX_CODE == 'F'), 'SEX_CODE'] = 1.0
  X = X.to_numpy(dtype=np.float64)

  y = df.LENGTH_OF_STAY
  y = y.to_numpy()
  if outcome == "LOS":
    return X, y
  elif outcome == ">12h":
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
  elif outcome == ">1d":
    y[y > 1] = 1
    y[y <= 1] = 0
  elif outcome == "nranges":
    assert nranges is not None, "Please specify 'nranges'"
    # TODO: Is this right to encode range like this?
    for i in range(1, len(nranges)):
      y[y >= nranges[i-1] & y < nranges[i]] = i - 1

  return X, y


# Perform train-test split (treat the test set as validation set)
def gen_train_val_test(X, y, val_pct=0.2, test_pct=0.2):
  X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_pct, random_state=SEED)
  X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_pct, random_state=SEED)
  return X_train, X_val, X_test, y_train, y_val, y_test


def standardize(X_train, X_test=None):
  scaler = StandardScaler().fit(X_train)
  if X_test is None:
    return scaler.transform(X_train)
  return scaler.transform(X_test)


# Run models
def predict_los(Xtrain, ytrain, Xtest, ytest, model='reg'):

  return