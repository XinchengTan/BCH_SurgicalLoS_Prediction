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

import pandas as pd
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor


