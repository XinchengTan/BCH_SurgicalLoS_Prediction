import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge, RidgeCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier
import shap

from . import globals
from .c4_model_eval import MyScorer


def init_perf_df(md, scorers):
  # acc, acc_1nnt_tol, overpred_rate, underpred_rate
  md2Classifier = {globals.LGR: LogisticRegression(), globals.SVC: SVC(), globals.KNN: KNeighborsClassifier(),
                   globals.DTCLF: DecisionTreeClassifier(), globals.RMFCLF: RandomForestClassifier(),
                   globals.GBCLF: GradientBoostingClassifier(), globals.XGBCLF: XGBClassifier()}
  params = md2Classifier.get(md, [])
  perf_df = pd.DataFrame(columns=params)
  # top K most important features [optional, since some models do not have this]
  return


def add_perf_row(perf_df, new_entry):

  return perf_df


def make_perf_row(clf, X, y, fold):

  return
