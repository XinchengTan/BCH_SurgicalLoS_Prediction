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
    #     clf = LogisticRegression(C=0.01, class_weight=cls_weight, max_iter=500, random_state=0)
    clf = LogisticRegression(random_state=SEED, C=0.03, class_weight=cls_weight, max_iter=500)
  elif model == LGR_L1:
    clf = LogisticRegression(random_state=SEED, C=0.3, class_weight=cls_weight, penalty='l1', solver='saga',
                             max_iter=500)
  elif model == LGR_L12:
    clf = LogisticRegression(random_state=SEED, C=0.1, l1_ratio=0.3, class_weight=cls_weight, penalty='elasticnet',
                             solver='saga', max_iter=500)
  elif model == LGRCV:
    clf = LogisticRegressionCV(Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1, 3], class_weight=cls_weight, max_iter=500,
                               random_state=SEED, cv=5)
  elif model == PR:
    clf = RegressionBasedClassifier(PR, alpha=0.01, max_iter=300)
  elif model == SVCLF:
    clf = SVC(gamma='scale', C=3, class_weight=cls_weight, probability=True)
  elif model == SVC_LINEAR:
    clf = LinearSVC(C=1, class_weight=cls_weight, random_state=SEED)
  elif model == KNN:
    #clf = KNeighborsClassifier(weights='uniform', p=2, n_neighbors=25, leaf_size=20, metric='minkowski')
    clf = KNeighborsClassifier(n_neighbors=45)
    # clf = KNeighborsClassifier(weights='uniform', p=1, n_neighbors=62, leaf_size=440, algorithm='ball_tree')
  elif model == KNNCV:
    clf = KNeighborsClassifierCV({'n_neighbors': np.arange(5, 51, 5)})
  elif model == DTCLF:
    # {'splitter': 'best', 'min_samples_split': 116, 'min_samples_leaf': 8, 'max_leaf_nodes': None, 'max_features': 392,
    # 'max_depth': 11, 'ccp_alpha': 0.001}
    clf = DecisionTreeClassifier(random_state=SEED, max_depth=11, splitter='best', min_samples_split=116,
                                 min_samples_leaf=8, max_leaf_nodes=None, max_features=392, ccp_alpha=0.001,
                                 class_weight=cls_weight)
  elif model == RMFCLF:
    clf = RandomForestClassifier(random_state=SEED, n_estimators=50, min_samples_split=62, min_samples_leaf=4,
                                 max_samples=0.6, max_leaf_nodes=90, max_features=322, max_depth=20,
                                 class_weight=cls_weight)
    #clf = RandomForestClassifier(random_state=0, max_depth=20, class_weight=cls_weight)
    # clf = RandomForestClassifier(random_state=SEED, n_estimators=100, min_samples_split=2, min_samples_leaf=6,
    #                              max_samples=0.6, max_leaf_nodes=50, max_features=362, max_depth=12,
    #                              class_weight=cls_weight)
  elif model == GBCLF:
    clf = GradientBoostingClassifier(random_state=0, max_depth=3)
  elif model == XGBCLF:
    clf = XGBClassifier(max_depth=6, learning_rate=0.03, n_estimators=300, subsample=0.95, min_child_weight=0.1,
                        reg_lambda=0.1, gamma=3, colsample_bytree=0.7, colsample_bylevel=0.8, num_class=len(NNT_CLASSES),
                        random_state=SEED, use_label_encoder=False, eval_metric='mlogloss', objective='multi:softmax', )
    # clf = XGBClassifier(max_depth=6, learning_rate=0.03, n_estimators=200, random_state=SEED, use_label_encoder=False,
    #                     eval_metric='mlogloss', objective='multi:softmax', num_class=MAX_NNT+2,
    #                     subsample=1, min_child_weight=2, reg_lambda=1, gamma=0.1, colsample_bytree=0.9)
  elif model == BAGCLF:
    clf = BaggingClassifier(n_estimators=200, max_samples=0.4, max_features=0.4, bootstrap_features=False,
                            random_state=SEED)
  elif model == GAUSSNB:
    clf = GaussianNB()
  elif model == ADABOOST:
    clf = AdaBoostClassifier(n_estimators=100, random_state=SEED)
  elif model == EXTREECLF:
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=SEED)
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
def get_model_binclf(model, cls_weight=None):
  if model == LGR:
    clf = LogisticRegression(random_state=SEED, C=0.03, class_weight=cls_weight, max_iter=500)
  elif model == LGR_L1:
    clf = LogisticRegression(random_state=SEED, C=0.3, class_weight=cls_weight, penalty='l1', solver='saga',
                             max_iter=500)
  elif model == LGR_L12:
    clf = LogisticRegression(random_state=SEED, C=0.1, l1_ratio=0.3, class_weight=cls_weight, penalty='elasticnet',
                             solver='saga', max_iter=500)
  elif model == PR:
    clf = RegressionBasedClassifier(PR, alpha=0.01, max_iter=300)
  elif model == SVCLF:
    clf = SVC(gamma='scale', C=3, class_weight=cls_weight, probability=True)
  elif model == SVC_LINEAR:
    clf = CalibratedClassifierCV(LinearSVC(C=1, class_weight=cls_weight, random_state=SEED))
  elif model == KNN:
    clf = KNeighborsClassifier(n_neighbors=45)
  elif model == DTCLF:
    clf = DecisionTreeClassifier(random_state=SEED, max_depth=11, splitter='best', min_samples_split=116,
                                 min_samples_leaf=8, max_leaf_nodes=None, max_features=392, ccp_alpha=0.001,
                                 class_weight=cls_weight)
  elif model == RMFCLF:
    clf = RandomForestClassifier(random_state=SEED, n_estimators=50, min_samples_split=62, min_samples_leaf=4,
                                 max_samples=0.6, max_leaf_nodes=90, max_features=322, max_depth=20,
                                 class_weight=cls_weight)
  elif model == GBCLF:
    clf = GradientBoostingClassifier(random_state=0, max_depth=3)
  elif model == XGBCLF:
    clf = XGBClassifier(max_depth=6, learning_rate=0.03, n_estimators=300, subsample=0.95, min_child_weight=0.1,
                        reg_lambda=0.1, gamma=3, colsample_bytree=0.7, colsample_bylevel=0.8, num_class=2,
                        random_state=SEED, use_label_encoder=False, eval_metric='mlogloss', objective='multi:softmax',)
  elif model == BAGCLF:
    clf = BaggingClassifier(n_estimators=200, max_samples=0.4, max_features=0.4, bootstrap_features=False,
                            random_state=SEED)
  elif model == GAUSSNB:
    clf = GaussianNB()
  elif model == CNB:
    clf = ComplementNB(norm=True, alpha=1)
  elif model == MNB:
    clf = MultinomialNB(alpha=3)
  elif model == ADABOOST:
    clf = AdaBoostClassifier(n_estimators=100, random_state=SEED)
  elif model == EXTREECLF:
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=SEED)
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
