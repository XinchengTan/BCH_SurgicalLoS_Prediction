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
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
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
def gen_Xy(df, outcome="los", cols=None):
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
    reg = LinearRegression().fit(Xtrain, ytrain)
  elif model == 'ridgecv':
    reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(Xtrain, ytrain)
  elif model == 'dt':
    reg = DecisionTreeRegressor(random_state=0, max_depth=4).fit(Xtrain, ytrain)
  elif model == 'rmf':
    reg = RandomForestRegressor(max_depth=5, random_state=0).fit(Xtrain, ytrain)
  elif model == 'gb':
    reg = GradientBoostingRegressor(random_state=0, max_depth=2).fit(Xtrain, ytrain)
  else:
    raise ValueError("Model %s not supported!" % model)

  if eval:
    pred_train, pred_val = reg.predict(Xtrain), reg.predict(Xtest)
    train_mse = mean_squared_error(ytrain, pred_train)
    val_mse = mean_squared_error(ytest, pred_val)
    print("%s:" % globals.model2name[model])
    print("R-squared (training set): ", reg.score(Xtrain, ytrain))
    print("R-squared (validation set): ", reg.score(Xtest, ytest))
    print("MSE (training set): ", train_mse, "RMSE: ", np.sqrt(train_mse))
    print("MSE (validation set): ", val_mse, "RMSE: ", np.sqrt(val_mse))
  return reg


def run_classifier(Xtrain, ytrain, Xtest, ytest, model='lr'):

  pass


def gen_feature_importance(model, mdabbr, ftrs=globals.FEATURE_COLS, pretty_print=False):
  sorted_frts = [(x, y) for y, x in sorted(zip(model.feature_importances_, ftrs), reverse=True, key=lambda p: p[0])]
  if pretty_print:
    print("\n" + globals.model2name[mdabbr] + ":")
    c = 1
    for x, y in sorted_frts:
      print("{c}.{ftr}:  {score}".format(c=c, ftr=x, score=round(y, 4)))
      c += 1
  return sorted_frts


def predict_los(Xtrain, ytrain, Xtest, ytest, outcome='los', model=None, isTest=False):
  model2trained = {}

  if model is None:  # run all
    if outcome == globals.LOS:
      for md in globals.model2name.keys():
        md = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=md)
        model2trained[model] = md

    elif outcome == globals.NNT:
      for md, md_name in globals.model2name.items():
        reg = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=md, eval=True)
        model2trained[md] = reg
        # predict and round to the nearest int
        pred_train, pred_test = np.rint(reg.predict(Xtrain)), np.rint(reg.predict(Xtest))
        # bucket them into finite number of classes
        pred_train[pred_train > globals.MAX_NNT] = globals.MAX_NNT + 1
        pred_test[pred_test > globals.MAX_NNT] = globals.MAX_NNT + 1
        # confusion matrix
        labels = [str(i) for i in range(globals.MAX_NNT+2)]
        labels[-1] = '%s+' % globals.MAX_NNT
        confmat_train = confusion_matrix(ytrain, pred_train, labels=np.arange(0, globals.MAX_NNT+2, 1), normalize='true')
        confmat_test = confusion_matrix(ytest, pred_test, labels=np.arange(0, globals.MAX_NNT+2, 1), normalize='true')
        print("Accuracy (training): ", accuracy_score(ytrain, pred_train, normalize=True))
        print("Accuracy (validation): ", accuracy_score(ytest, pred_test, normalize=True))
        # plot
        figs, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 21))
        sn.set(font_scale=1.3)  # for label size
        # sn.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
        sn.heatmap(confmat_train, fmt=".2%", cmap=sn.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
                   annot=True, annot_kws={"size": 16}, ax=axs[0])  # font size
        axs[0].set_title("Confusion Matrix (%s - training)" % md_name, fontsize=20, y=1.01)
        axs[0].set_xlabel("Predicted outcome", fontsize=16)
        axs[0].set_ylabel("True outcome", fontsize=16)
        sn.heatmap(confmat_test, fmt=".2%", cmap='rocket_r', annot=True, annot_kws={"size": 16}, ax=axs[1])
        axs[1].set_title("Confusion Matrix (%s - test)" % md_name if isTest else "Confusion Matrix (%s - validation)" % md_name,
                         fontsize=18, y=1.01)
        axs[1].set_xlabel("Predicted outcome", fontsize=16)
        axs[1].set_ylabel("True outcome", fontsize=16)
        plt.show()

  else:
    trained_model = run_regression_model(Xtrain, ytrain, Xtest, ytest, model=model)
    model2trained[model] = trained_model

  return model2trained

