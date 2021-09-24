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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

from . import globals

SEED = 998

# TODO: add data cleaning, e.g. check for negative LoS and remove it

# Generate data matrix X
def gen_Xy(df, outcome="los", nranges=None, cols=None):
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
            'Uncategorized', 'Urogenital', 'NUM_OF_NIGHTS']
  X = df[cols]
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
  X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_pct, random_state=SEED)
  X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_pct, random_state=SEED)
  return X_train, X_val, X_test, y_train, y_val, y_test


def standardize(X_train, X_test=None):
  scaler = StandardScaler().fit(X_train)
  if X_test is None:
    return scaler.transform(X_train)
  return scaler.transform(X_test)


def run_regression_model(Xtrain, ytrain, Xtest, ytest, model='lr', eval=True):
  if model == 'lr':
    model = LinearRegression().fit(Xtrain, ytrain)
  elif model == 'ridgecv':
    model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(Xtrain, ytrain)
  elif model == 'dt':
    model = DecisionTreeRegressor(random_state=0, max_depth=4).fit(Xtrain, ytrain)
  elif model == 'rmf':
    model = RandomForestRegressor(max_depth=5, random_state=0).fit(Xtrain, ytrain)
  elif model == 'gb':
    model = GradientBoostingRegressor(random_state=0, max_depth=2).fit(Xtrain, ytrain)
  else:
    raise ValueError("Model %s not supported!" % model)

  if eval:
    pred_train, pred_val = model.predict(Xtrain), model.predict(Xtest)
    train_mse = mean_squared_error(ytrain, pred_train)
    val_mse = mean_squared_error(ytest, pred_val)
    print("%s:" % globals.model2name[model])
    print("R-squared (training set): ", model.score(Xtrain, ytrain))
    print("R-squared (validation set): ", model.score(Xtest, ytest))
    print("MSE (training set): ", train_mse, "RMSE: ", np.sqrt(train_mse))
    print("MSE (validation set): ", val_mse, "RMSE: ", np.sqrt(val_mse))
  return model


def run_classifier(Xtrain, ytrain, Xtest, ytest, model='lr'):

  pass


def gen_confusion_matrix(y_true, y_pred, labels=None):

  return


def predict_los(Xtrain, ytrain, Xtest, ytest, outcome='los', model=None, isTest=False):
  model2trained = {}

  if model is None:  # run all
    if outcome == globals.LOS:
      for md in globals.model2name.keys():
        md = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=md)
        model2trained[model] = md

    elif outcome == globals.NNT:
      for md in globals.model2name.keys():
        md = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=md, eval=False)
        model2trained[model] = md
        # predict and round to the nearest int
        pred_train, pred_test = np.rint(model.predict(Xtrain)), np.rint(model.predict(Xtest))
        # bucket them into finite number of classes
        pred_train[pred_train > globals.MAX_NNT] = globals.MAX_NNT + 1
        pred_test[pred_test > globals.MAX_NNT] = globals.MAX_NNT + 1
        # confusion matrix
        confmat_train = confusion_matrix(ytrain, pred_train, labels=np.arange(0, globals.MAX_NNT+2, 1))
        confmat_test = confusion_matrix(ytest, pred_test, labels=np.arange(0, globals.MAX_NNT+2, 1))
        # plot
        figs, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 7))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(confmat_train, annot=True, annot_kws={"size": 16}, ax=axs[0])  # font size
        axs[0].set_title("Confusion Matrix (training set)")
        sn.heatmap(confmat_test, annot=True, annot_kws={"size": 16}, ax=axs[0])  # font size
        axs[1].set_title("Confusion Matrix (test set)" if isTest else "Confusion Matrix (validation set)")
        plt.show()

  else:
    trained_model = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=model)
    model2trained[model] = trained_model

  return model2trained

