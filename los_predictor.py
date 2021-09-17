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
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor


# Generate data matrix X
def gen_Xy(df):
  y = df.LENGTH_OF_STAY

  cols = ['SEX_CODE', 'AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE',
          'PROC_DECILE', 'Cardiovascular', 'Digestive',
          'Endocrine', 'Genetic', 'Hematologic', 'Immunologic', 'Infectious',
          'Mental', 'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic',
          'Nutrition', 'Optic', 'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin',
          'Uncategorized', 'Urogenital']
  X = df[cols]
  X['SEX_CODE'] = 0 if X['SEX_CODE'] == "F" else 1
  X = X.to_numpy()
  y = y.to_numpy()

  return X, y


# Perform train-test split (treat the test set as validation set)
def gen_train_test(X, y, test_pct=0.3):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, random_state=98)
  return X_train, X_test, y_train, y_test


# Run models
def predict_los(Xtrain, ytrain, Xtest, ytest, model='reg'):
  
  return