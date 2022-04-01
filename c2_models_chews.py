from globals import *
from c1_data_preprocessing import Dataset
from c2_models import *
from c4_model_perf import MyScorer


# ---------------------------------------- Respiratory Decline prediction ----------------------------------------
def get_model_respir_decline(model, cls_weight='balanced'):
  if model == LGR:
    clf = LogisticRegression(C=0.03, class_weight=cls_weight, max_iter=500, random_state=0)
  elif model == LGRCV:
    clf = LogisticRegressionCV(Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1, 3], class_weight=cls_weight, max_iter=500,
                               random_state=0, cv=5)
  elif model == PR:
    clf = RegressionBasedClassifier(PR, alpha=0.01, max_iter=300)
  elif model == SVCLF:
    clf = SVC(gamma='scale', class_weight=cls_weight, probability=False)
  elif model == SVC_POLY3:
    clf = SVC(gamma='scale', kernel='poly', degree=3, class_weight=cls_weight, probability=False)
  elif model == SVC_POLY4:
    clf = SVC(gamma='scale', kernel='poly', degree=4, class_weight=cls_weight, probability=False)
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
    # TODO: tune param 'scale_pos_weight'
    clf = XGBClassifier(max_depth=4, learning_rate=0.03, n_estimators=150, random_state=0, use_label_encoder=False,
                        eval_metric='mlogloss', scale_pos_weight=90)
  elif model == ORDCLF_LOGIT:
    clf = OrdinalClassifier(distr='logit', solver='bfgs', disp=True)
  elif model == ORDCLF_PROBIT:
    clf = OrdinalClassifier(distr='probit', solver='bfgs', disp=True)
  elif model == BAL_BAGCLF:
    clf = BalancedBaggingClassifier(random_state=0, n_estimators=100, n_jobs=-1)
  else:
    raise NotImplementedError("Model %s is not supported!" % model)
  return SafeOneClassWrapper(clf)


# --------------------------------------- Cardiovascular Decline prediction ---------------------------------------
def get_model_cardio_decline(model, cls_weight='balanced'):
  if model == LGR:
    clf = LogisticRegression(C=0.03, class_weight=cls_weight, max_iter=500, random_state=0)
  elif model == LGRCV:
    clf = LogisticRegressionCV(Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1, 3], class_weight=cls_weight, max_iter=500,
                               random_state=0, cv=5)
  elif model == PR:
    clf = RegressionBasedClassifier(PR, alpha=0.01, max_iter=300)
  elif model == SVCLF:
    clf = SVC(gamma='scale', class_weight=cls_weight, probability=False)
  elif model == SVC_POLY3:
    clf = SVC(gamma='scale', kernel='poly', degree=3, class_weight=cls_weight, probability=False)
  elif model == SVC_POLY4:
    clf = SVC(gamma='scale', kernel='poly', degree=4, class_weight=cls_weight, probability=False)
  elif model == KNN:
    clf = KNeighborsClassifier(n_neighbors=45)
  elif model == KNNCV:
    clf = KNeighborsClassifierCV({'n_neighbors': [5, 10, 15, 20, 30, 40]})
  elif model == DTCLF:
    clf = DecisionTreeClassifier(random_state=0, max_depth=4, class_weight=cls_weight)
  elif model == RMFCLF:
    clf = RandomForestClassifier(random_state=0, max_depth=10, class_weight=cls_weight)
  elif model == GBCLF:
    clf = GradientBoostingClassifier(random_state=0, max_depth=3)
  elif model == XGBCLF:
    clf = XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, random_state=0, use_label_encoder=False,
                        eval_metric='mlogloss', scale_pos_weight=98)
  elif model == ORDCLF_LOGIT:
    clf = OrdinalClassifier(distr='logit', solver='bfgs', disp=True)
  elif model == ORDCLF_PROBIT:
    clf = OrdinalClassifier(distr='probit', solver='bfgs', disp=True)
  elif model == BAL_BAGCLF:
    clf = BalancedBaggingClassifier(random_state=0)
  else:
    raise NotImplementedError("Model %s is not supported!" % model)
  return SafeOneClassWrapper(clf)


# --------------------------------------- Neurologic Decline prediction ---------------------------------------
def get_model_neuro_decline(model, cls_weight='balanced'):
  if model == LGR:
    clf = LogisticRegression(C=0.03, class_weight=cls_weight, max_iter=500, random_state=0)
  elif model == LGRCV:
    clf = LogisticRegressionCV(Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1, 3], class_weight=cls_weight, max_iter=500,
                               random_state=0, cv=5)
  elif model == PR:
    clf = RegressionBasedClassifier(PR, alpha=0.01, max_iter=300)
  elif model == SVCLF:
    clf = SVC(gamma='scale', class_weight=cls_weight, probability=False)
  elif model == SVC_POLY3:
    clf = SVC(gamma='scale', kernel='poly', degree=3, class_weight=cls_weight, probability=False)
  elif model == SVC_POLY4:
    clf = SVC(gamma='scale', kernel='poly', degree=4, class_weight=cls_weight, probability=False)
  elif model == KNN:
    clf = KNeighborsClassifier(n_neighbors=45)
  elif model == KNNCV:
    clf = KNeighborsClassifierCV({'n_neighbors': [5, 10, 15, 20, 30, 40]})
  elif model == DTCLF:
    clf = DecisionTreeClassifier(random_state=0, max_depth=4, class_weight=cls_weight)
  elif model == RMFCLF:
    clf = RandomForestClassifier(random_state=0, max_depth=10, class_weight=cls_weight)
  elif model == GBCLF:
    clf = GradientBoostingClassifier(random_state=0, max_depth=3)
  elif model == XGBCLF:
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

