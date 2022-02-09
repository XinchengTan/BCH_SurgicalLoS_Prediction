import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import warnings

from imblearn.over_sampling import SMOTENC
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, make_scorer, plot_roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor, Ridge, RidgeCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
# from statsmodels.miscmodels.ordinal_model import OrderedModel
from xgboost import XGBClassifier

from . import globals, c4_model_eval, c6_surgeon, utils
from . import c0_data_prepare as dp, c1_data_preprocessing as dpp
from .c1_data_preprocessing import Dataset
from .c3_ensemble import Ensemble
from .c4_model_eval import ModelPerf
from .c7_feature_engineering import FeatureEngineeringModifier, DecileGenerator
from . import utils_plot as pltutil


### Goal: hyper parameter tuning for models with different feature sets
### Input: preprocessed df with feature fully engineered

def get_args():
  parser = argparse.ArgumentParser(description='Hyperparam Tuning Script')
  # platform
  parser.add_argument('--platform', '-pf', default='aws', choices=['aws', 'local', 'gcp'], type=str)
  parser.add_argument('--device', '-d', default='cpu', choices=['cpu', 'cuda:0'], type=str)

  # dataset
  parser.add_argument('--cohort', default='all', type=str)
  parser.add_argument('--test_pct', default=0.2, type=float)
  parser.add_argument('--holdout_test', default=False, action='store_true')
  parser.add_argument('--skip_o2m', default=False, action='store_true')

  # features
  parser.add_argument('--age', default='disc', choices=['disc', 'cont', 'none'], type=str)
  parser.add_argument('--gender', default=False, action='store_true')
  parser.add_argument('--pproc', nargs='+')  # none, oh, dcl, cnt, sd, min, max ...
  parser.add_argument('--cpt', nargs='+')
  parser.add_argument('--cpt_grp', nargs='+')
  parser.add_argument('--ccsr', nargs='+')
  parser.add_argument('--med1', nargs='+')
  parser.add_argument('--med2', nargs='+')
  parser.add_argument('--med3', nargs='+')
  parser.add_argument('--os', nargs='+')  # only 'oh' is available
  # TODO: how to manage decile features

  # modeling
  parser.add_argument('--kfold', default=5, type=int)  # If 1, no cross validation
  parser.add_argument('--model', nargs='+')  # all, lgr, svc, knn, dt, rmf, xgb
  parser.add_argument('--ensemble', nargs='+')
  parser.add_argument('--md_fp', '--model_filepath', type=pathlib.Path)
  # TODO: Have an arg for filepath of model params?   arg for where results are saved?

  args = parser.parse_args()
  return args


if __name__ == '__main__':
  # read data

  import boto3

  client = boto3.client('s3')
  data_obj = client.get_object(Bucket='bch-data-cara', Key='test.csv')
  data_df = pd.read_csv(data_obj['Body'])

  for bucket in client.list_buckets()['Buckets']:
    print(bucket['Name'])


