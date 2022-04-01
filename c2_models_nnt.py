# Customized Model classes
import warnings

import numpy as np
import optuna
from collections import Counter
from tqdm import tqdm
from typing import Any, Dict, List

from globals import *
from c1_data_preprocessing import Dataset
from c2_models import *
from c4_model_perf import MyScorer


# ---------------------------------------- Multi-class NNT prediction ----------------------------------------
# Returns a model wrapped in SafeOneClassWrapper based on the requested type; model is not fitted
def get_model(model, cls_weight='balanced'):
  if model == LGR:
    clf = LogisticRegression(C=0.01, class_weight=cls_weight, max_iter=500, random_state=0)
  elif model == LGRCV:
    clf = LogisticRegressionCV(Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1, 3], class_weight=cls_weight, max_iter=500,
                               random_state=0, cv=5)
  elif model == PR:
    clf = RegressionBasedClassifier(PR, alpha=0.01, max_iter=300)
  elif model == SVCLF:
    clf = SVC(gamma='auto', class_weight=cls_weight, probability=False)
  elif model == KNN:
    clf = KNeighborsClassifier(n_neighbors=45)
  elif model == KNNCV:
    clf = KNeighborsClassifierCV({'n_neighbors': np.arange(5, 51, 5)})
  elif model == DTCLF:
    clf = DecisionTreeClassifier(random_state=0, max_depth=4, class_weight=cls_weight)
  elif model == RMFCLF:
    clf = RandomForestClassifier(random_state=0, max_depth=20, class_weight=cls_weight)
  elif model == GBCLF:
    clf = GradientBoostingClassifier(random_state=0, max_depth=3)
  elif model == XGBCLF:
    clf = XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=150, random_state=0, use_label_encoder=False,
                        eval_metric='mlogloss')
  elif model == ORDCLF_LOGIT:
    clf = OrdinalClassifier(distr='logit', solver='bfgs', disp=True)
  elif model == ORDCLF_PROBIT:
    clf = OrdinalClassifier(distr='probit', solver='bfgs', disp=True)
  elif model == BAL_BAGCLF:
    clf = BalancedBaggingClassifier(random_state=0)
  else:
    raise NotImplementedError("Model %s is not supported!" % model)

  return SafeOneClassWrapper(clf)


# ------------------------------------- Cohort-wise multi-class NNT prediction -------------------------------------
def get_model_by_cohort(model, cls_weight='balanced', cohort=None):
  # TODO: can add customization for each cohort later
  if model == LGR:
    clf = LogisticRegression(C=0.03, class_weight=cls_weight, max_iter=500, random_state=0)
    # clf = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], class_weight=cls_weight, max_iter=500, random_state=0,
    #                            cv=5)
  elif model == PR:
    clf = RegressionBasedClassifier(PR, alpha=0.01, max_iter=300)
  elif model == SVCLF:
    clf = SVC(gamma='auto', class_weight=cls_weight, probability=False)
  elif model == KNN:
    clf = KNeighborsClassifierCV({'n_neighbors': np.arange(4, 16)})
  elif model == DTCLF:
    clf = DecisionTreeClassifier(random_state=0, max_depth=4, class_weight=cls_weight)
  elif model == RMFCLF:
    clf = RandomForestClassifier(random_state=0, class_weight=cls_weight)
  elif model == GBCLF:
    clf = GradientBoostingClassifier(random_state=0, max_depth=3)
  elif model == XGBCLF:
    clf = XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, gamma=5, random_state=0,
                        use_label_encoder=False, eval_metric='mlogloss')
  elif model == ORDCLF_LOGIT:
    clf = OrdinalClassifier(distr='logit', solver='bfgs', disp=True)
  elif model == ORDCLF_PROBIT:
    clf = OrdinalClassifier(distr='probit', solver='bfgs', disp=True)
  elif model == BAL_BAGCLF:
    clf = BalancedBaggingClassifier(random_state=0)
  else:
    raise NotImplementedError("Model %s is not supported!" % model)

  return SafeOneClassWrapper(clf)


# ---------------------------------------- Binary-class NNT prediction ----------------------------------------
def get_model_binclf(model, cls_weight='balanced'):
  if model == LGR:
    clf = LogisticRegression(C=0.01, class_weight=cls_weight, max_iter=500, random_state=0)
  elif model == LGRCV:
    clf = LogisticRegressionCV(Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1, 3], class_weight=cls_weight, max_iter=500,
                               random_state=0, cv=5)
  elif model == PR:
    clf = RegressionBasedClassifier(PR, alpha=0.01, max_iter=300)
  elif model == SVCLF:
    clf = SVC(gamma='auto', class_weight=cls_weight, probability=False)
  elif model == KNN:
    clf = KNeighborsClassifier(n_neighbors=45)
  elif model == KNNCV:
    clf = KNeighborsClassifierCV({'n_neighbors': np.arange(5, 56, 10)})
  elif model == DTCLF:
    clf = DecisionTreeClassifier(random_state=0, max_depth=4, class_weight=cls_weight)
  elif model == RMFCLF:
    clf = RandomForestClassifier(random_state=0, max_depth=10, class_weight=cls_weight)
  elif model == GBCLF:
    clf = GradientBoostingClassifier(random_state=0, max_depth=3)
  elif model == XGBCLF:
    # TODO: enable 'scale_pos_weight' arg!!
    clf = XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, random_state=0, use_label_encoder=False,
                        eval_metric='mlogloss')
  elif model == ORDCLF_LOGIT:
    clf = OrdinalClassifier(distr='logit', solver='bfgs', disp=True)
  elif model == ORDCLF_PROBIT:
    clf = OrdinalClassifier(distr='probit', solver='bfgs', disp=True)
  elif model == BAL_BAGCLF:
    clf = BalancedBaggingClassifier(random_state=0)
  else:
    raise NotImplementedError("Model %s is not supported!" % model)

  return SafeOneClassWrapper(clf)
