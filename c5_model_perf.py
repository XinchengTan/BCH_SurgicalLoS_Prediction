import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score
import shap

from . import globals
from .c4_model_eval import MyScorer


def init_perf_df(md, clf_params):
  # acc, acc_1nnt_tol, overpred_rate, underpred_rate,

  # top K most important features [optional, since some models do not have this]
  return


def add_perf_row(perf_df, new_entry):

  return perf_df


def make_perf_row(clf, X, y):

  return
