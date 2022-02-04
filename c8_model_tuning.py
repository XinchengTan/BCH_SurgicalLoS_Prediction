from imblearn.over_sampling import SMOTENC
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, make_scorer, plot_roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from statsmodels.miscmodels.ordinal_model import OrderedModel
from xgboost import XGBClassifier

from . import globals, c4_model_eval, c6_surgeon, utils
from . import c1_data_preprocessing as dpp
from .c1_data_preprocessing import Dataset
from .c4_model_eval import ModelPerf
from .c3_ensemble import Ensemble
from . import utils_plot as pltutil


### Goal: hyper parameter tuning for models with different feature sets
### Input: preprocessed df