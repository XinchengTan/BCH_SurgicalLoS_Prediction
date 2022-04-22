import numpy as np
import torch
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB, CategoricalNB

from globals import *
from c1_data_preprocessing import Dataset
from c2_models_nnt import *
from c2_models_chews import *
from c3_ensemble import *
from c4_model_perf import MyScorer


# -------------------------------------- Train models on k Datasets (Multi-class) --------------------------------------
def train_model_all_ktrials(models, k_datasets: Dict[Any, Dataset], cls_weight,
                            train_sda_only=False, train_surg_only=False, train_care_class=None) -> Dict:
  models = [LGR, KNN, RMFCLF, XGBCLF] if models is None else models  # GBCLF,
  k_model_dict = {}
  for k, dataset_k in tqdm(k_datasets.items()):
    # Fit models
    Xtrain, ytrain = dataset_k.get_Xytrain_by_case_key(dataset_k.train_case_keys, care_class=train_care_class,
                                                       sda_only=train_sda_only, surg_only=train_surg_only)
    model_dict = {}
    for md in models:
      print('md=', md)
      if dataset_k.outcome == NNT:
        if md != SUPER_LEARNER:
          clf = get_model(md, cls_weight=cls_weight)
        else:
          print('Using super learner!')
          clf = SuperLearner(get_default_base_models(), LGR, base_fitted=False)
      elif dataset_k.outcome in BINARY_NNT_SET:
        clf = get_model_binclf(md, cls_weight=cls_weight)
      elif dataset_k.outcome == RESPIR_DECLINE:
        clf = get_model_respir_decline(md, cls_weight=cls_weight)
      elif dataset_k.outcome == CARDIO_DECLINE:
        clf = get_model_cardio_decline(md, cls_weight=cls_weight)
      elif dataset_k.outcome == NEURO_DECLINE:
        clf = get_model_neuro_decline(md, cls_weight=cls_weight)
      else:
        warnings.warn(f'Outcome "{dataset_k.outcome}" is not supported yet! Skipped training')
        continue
      clf.fit(Xtrain, ytrain)
      model_dict[md] = clf
    k_model_dict[k] = model_dict
  return k_model_dict


# ------------------------------------- Train models on k Datasets (Binary class) --------------------------------------
# Train models for all binary outcomes across k trials/folds
def train_model_all_ktrials_binclf(models, bin_kt_datasets: Dict[str, Dict[int, Dataset]], cls_weight,
                                   train_sda_only=False, train_surg_only=False, train_care_class=None):
  bin_k_model_dict = defaultdict(dict)
  for bin_nnt, kt_datasets in tqdm(bin_kt_datasets.items(), desc='Binary Outcomes'):
    print('Outcome: ', bin_nnt)
    bin_k_model_dict[bin_nnt] = train_model_all_ktrials(models, k_datasets=kt_datasets, cls_weight=cls_weight,
                                                        train_sda_only=train_sda_only, train_surg_only=train_surg_only,
                                                        train_care_class=train_care_class)
  return bin_k_model_dict


def get_refit_val(scorers, refit):
  if (len(scorers) == 1) or (type(refit) == str):
    return refit

  if SCR_AUC in scorers:
    return SCR_AUC
  return SCR_ACC


# ------------------------------------- Hyperparameter Tuning with OptunaSearchCV -------------------------------------
def tune_model_optuna(md, X, y, kfold, scorers, refit=True):
  minority_size = min(Counter(y).values())
  print('Minority class size: ', minority_size)
  if md == KNN:
    clf = KNeighborsClassifier()
    # param_space = {
    #   'algorithm': optuna.distributions.CategoricalDistribution(['ball_tree', 'kd_tree', 'brute']),
    #   'leaf_size': optuna.distributions.DiscreteUniformDistribution(10, minority_size, 10),
    #   'n_neighbors': optuna.distributions.DiscreteUniformDistribution(5, minority_size, minority_size // 10 + 1),
    #   'p': optuna.distributions.IntUniformDistribution(1, 3),
    #   'weights': optuna.distributions.CategoricalDistribution(['uniform', 'distance'])
    # }
    param_space = {
      'n_neighbors': optuna.distributions.IntUniformDistribution(5, minority_size, minority_size // 10 + 1),
      'p': optuna.distributions.IntUniformDistribution(1, 3),
    }
  else:
    raise NotImplementedError
  refit = get_refit_val(scorers, refit)
  optuna_search = optuna.integration.OptunaSearchCV(clf, param_space, cv=kfold, refit=refit, n_trials=50, verbose=2,
                                                    scoring=scorers)
  optuna_search.fit(X, y)

  return optuna_search


# ------------------------------------- Hyperparameter Tuning with RandomSearchCV -------------------------------------
def tune_model_randomSearch(md, X, y, kfold, scorers, args, n_iters=20, refit=False, cls_weight=None, calibrate=False,
                            use_gpu=False):
  binary_cls = len(set(y)) < 3
  clf, param_space = gen_model_param_space(md, X, y, scorers=scorers, kfold=kfold, use_gpu=use_gpu)

  # Adjust gpu n_jobs accordingly
  if args is None:
    n_jobs = -1
  else:
    n_jobs = int(args.n_jobs) if use_gpu and (torch.cuda.is_available()) else -1  # avoid mem out when tuning on gpu

  if count_total_candidates(param_space) < n_iters:
    print('Default to GridSearchCV, given #candidates < n_iters')
    search_cv = GridSearchCV(estimator=clf, param_grid=param_space, n_jobs=n_jobs,
                             cv=KFold(n_splits=kfold, shuffle=True, random_state=SEED),
                             refit=refit, scoring=MyScorer.get_scorer_dict(scorers, binary_cls=binary_cls),
                             return_train_score=True, verbose=4)
  else:
    search_cv = RandomizedSearchCV(estimator=clf, param_distributions=param_space, n_iter=n_iters, n_jobs=n_jobs,
                                   cv=KFold(n_splits=kfold, shuffle=True, random_state=SEED), random_state=SEED,
                                   refit=refit, scoring=MyScorer.get_scorer_dict(scorers, binary_cls=binary_cls),
                                   return_train_score=True, verbose=4)

  # define sample weights, based on class_weight scheme
  if md != KNN:
    sample_weights = gen_sample_weights(y, cls_weight=cls_weight)
    search_cv.fit(X, y, sample_weight=sample_weights)
  else:
    search_cv.fit(X, y)
  return search_cv


# ------------------------------------- Hyperparameter Tuning with GridSearchCV -------------------------------------
def tune_model_gridSearch(md, X, y, scorers, args, kfold=5, cls_weight=None, refit=False, use_gpu=False):
  binary_cls = len(set(y)) < 3
  clf, param_space = gen_model_param_space(md, X, y, scorers, kfold=kfold, use_gpu=use_gpu)
  # Adjust gpu n_jobs accordingly
  n_jobs = int(args.n_jobs) if use_gpu and (torch.cuda.is_available()) else -1  # avoid mem out when tuning on gpu

  # Use GridSearchCV for hyperparameter tuning
  grid_search = GridSearchCV(estimator=clf, param_grid=param_space, n_jobs=n_jobs,
                             cv=KFold(n_splits=kfold, shuffle=True, random_state=SEED),
                             refit=refit, scoring=MyScorer.get_scorer_dict(scorers, binary_cls=binary_cls),
                             return_train_score=True, verbose=2)
  # define sample weights, based on class_weight scheme
  sample_weights = gen_sample_weights(y, cls_weight=cls_weight)
  grid_search.fit(X, y, sample_weight=sample_weights)
  return grid_search


# TODO: add arg for outcome
def gen_model_param_space(md, X, y, scorers, kfold=5, use_gpu=False):
  n_frts = X.shape[1]
  minority_size = min(Counter(y).values())
  print("Minority-class size: ", minority_size)

  min_samples_split_max = int(minority_size * (1 - 1 / kfold))
  if md == LGR:
    param_space = {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                   }
    clf = LogisticRegression(random_state=SEED, max_iter=1000)
  elif md == LGR_L1:
    param_space = {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                   }
    clf = LogisticRegression(random_state=SEED, penalty='l1', solver='saga', max_iter=1000)
  elif md == LGR_L12:
    param_space = {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                   'l1_ratio': [0, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99, 1],
                   }
    clf = LogisticRegression(random_state=SEED, penalty='elasticnet', solver='saga', max_iter=1000)
  elif md == CATNB:
    param_space = {'alpha': [0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 2, 3, 4, 6, 8, 10, 12, 15]}
    clf = CategoricalNB()
  elif md == CNB:
    param_space = {'alpha': [0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 2, 3, 4, 6, 8, 10, 12, 15],
                   'norm': [False, True]}
    clf = ComplementNB()
  elif md == MNB:
    param_space = {'alpha': [0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 2, 3, 4, 6, 8, 10, 12, 15]}
    clf = MultinomialNB()
  elif md == SVCLF:
    clf = SVC(random_state=SEED, kernel='rbf')
    param_space = {'C': [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                   'gamma': ['scale'] + list({1 / n_frts, 1, 0.3, 0.1, 0.03, 0.01, 0.001}),
                   }
  elif md == SVC_POLY:
    clf = SVC(random_state=SEED, kernel='poly')
    param_space = {'C': [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                   'gamma': ['scale'] + list({1 / n_frts, 1, 0.3, 0.1, 0.03, 0.01, 0.001}),
                   'coef0': [0, 0.001, 0.01, 0.1, 1],
                   'degree': [2, 3, 4, 5],
                   }
  elif md == KNN:
    clf = KNeighborsClassifier(algorithm='auto', n_jobs=-1)
    param_space = {
      'leaf_size': list(range(20, 51, 5)) + [100, 150, 200, 300],
      'n_neighbors': list(range(5, 101, 5)),
      'p': [1, 2],
      'weights': ['uniform', 'distance'],
      'metric': ['chebyshev', 'minkowski']
    }
  elif md == RMFCLF:
    clf = RandomForestClassifier(random_state=SEED)
    param_space = {
      'max_depth': [None] + list(range(2, 21, 2)),
      'max_features': sorted(set(list(range(2, n_frts // 2 + 1, n_frts // 20 + 1)) + [int(np.sqrt(n_frts))])),
      'max_leaf_nodes': [None] + list(range(5, 101, 5)),
      'max_samples': np.arange(0.1, 1, 0.1),
      'min_samples_leaf': [1] + [i for i in range(2, 17, 2)],
      'min_samples_split': list(range(2, min_samples_split_max, 10)),
      'n_estimators': [20, 30, 40, 50, 80, 100, 200, 300, 400]
    }
    if not param_space['min_samples_split']:
      raise ValueError("min_samples_split cannot < 2!")
  elif md == XGBCLF:
    num_classes = len(set(y))
    objective = 'multi:softmax' if num_classes > 2 else 'binary:logistic'
    if use_gpu and torch.cuda.is_available():
      clf = XGBClassifier(random_state=SEED, eval_metric='mlogloss', use_label_encoder=False, objective=objective,
                          num_class=num_classes, tree_method='gpu_hist', gpu_id=0)
    else:
      clf = XGBClassifier(random_state=SEED, eval_metric='mlogloss', use_label_encoder=False, objective=objective,
                          num_class=num_classes)
    if n_frts > 500:
      colsample_bytree_range = np.arange(0.3, 1.01, 0.1)
    else:
      colsample_bytree_range = np.arange(0.8, 1.01, 0.05)
    param_space = {
      'n_estimators': [50, 100, 150, 200, 300],
      'learning_rate': [0.003, 0.01, 0.03, 0.1, 0.3, 1],
      'subsample': np.arange(0.8, 1.01, 0.05),
      'colsample_bytree': colsample_bytree_range,
      'colsample_bylevel': [0.4, 0.8, 0.9, 1],
      'max_depth': [3, 4, 5, 6, 7],
      'gamma': [0, 0.1, 0.3, 1, 3, 5],
      'min_child_weight': [0.01, 0.1, 0.3, 0.6, 1, 2],
      'lambda': [0, 0.01, 0.1, 1],
    }
    # Tune unbalanced binary class weight parameter
    if num_classes == 2:
      unbalanced_ratio = int((len(y) - minority_size) / minority_size)
      step = np.round(np.sqrt(unbalanced_ratio))
      param_space['scale_pos_weight'] = np.arange(1, unbalanced_ratio + 1, step)
      if unbalanced_ratio not in param_space['scale_pos_weight']:
        param_space['scale_pos_weight'] = np.append(param_space['scale_pos_weight'], unbalanced_ratio)
  elif md == BAGCLF:
    print('tuning bagging clf')
    clf = BaggingClassifier(random_state=SEED)  #, n_jobs=30
    param_space = {
      # 'base_estimator__max_depth': [1, 2, 3, 4, 5],
      'n_estimators': [20, 40, 60, 80, 100, 120, 150, 200, 300],
      'max_samples': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
      'max_features': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
      'bootstrap_features': [True, False]
    }
  elif md == CATBOOST:
    # TODO: finish this
    clf = CatBoostClassifier()
    param_space = {
      'learning_rate': [0.003, 0.01, 0.03, 0.1, 0.3, 1],
      'depth': np.arange(2, 9)
    }
  elif md == GBCLF:
    # TODO: what's validation fraction?
    clf = GradientBoostingClassifier(random_state=SEED, validation_fraction=0.15, n_iter_no_change=3)
    param_space = {
      'learning_rate': [0.001, 0.03, 0.01, 0.3, 0.1, 0.3],
      'loss': 'deviance',
      'max_depth': [None] + list(range(2, 21, 2)),
      'max_features': [None] + list(range(2, 1 + n_frts // 2, 2)),
      'max_leaf_nodes': [None] + list(range(5, )),
      'min_samples_leaf': [1, 2, 3, 4, 5],
      'min_samples_split': list(range(2, min_samples_split_max, 3)),
      'n_estimators': [20, 30, 40, 50, 80, 100, 200, 300, 400]
    }
    if not param_space['min_samples_split']:
      raise ValueError("min_samples_split cannot < 2!")
  elif md == DTCLF:
    clf = DecisionTreeClassifier(random_state=SEED)
    param_space = {
      'ccp_alpha': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
      'max_depth': [None] + list(range(2, 21, 1)),
      'max_features': ['sqrt', 'log2', None, 0.5, 0.6, 0.7, 0.8, 0.9],
      'max_leaf_nodes': [None] + list(range(5, 101, 5)),
      'min_samples_leaf': [1] + [i for i in range(2, 17, 2)],
      'min_samples_split': list(range(2, min_samples_split_max, 3)),
      'splitter': ['best', 'random'],
    }
  else:
    raise NotImplementedError("Model %s is not supported!" % md)

  return clf, param_space


def count_total_candidates(param_space: Dict):
  if len(param_space) == 0:
    return 0
  count = 1
  for k, v in param_space.items():
    count *= len(v)
  return count


# Generate sample weights to use in model.fit(sample_weight=...)
def gen_sample_weights(y, cls_weight=None):
  if cls_weight is None:
    return np.ones_like(y)

  if type(cls_weight) == float:
    cls_weight = flex_log_class_weight(y, mu=cls_weight)
  elif cls_weight != 'balanced':
    raise NotImplementedError(f'Got an invalid cls_weight: {cls_weight}!')
  # if 'balanced', class_weight = n_samples / (n_classes * np.bincount(y))
  print('[c2_models_train_tune] class_weight: ', cls_weight)

  sample_weights = class_weight.compute_sample_weight(class_weight=cls_weight, y=y)
  return sample_weights


# Log-based smooth class weight for imbalanced class
def flex_log_class_weight(y, mu=0.15):
  label2count = Counter(y)
  total = np.sum(list(label2count.values()))
  class_weight = dict()

  for cls, cls_cnt in label2count.items():
    score = np.log(mu * total / float(cls_cnt))
    class_weight[cls] = max(1.0, score)  # todo: why 1.0 instead of a smaller number?

  return class_weight


## XGB other params:
# 'reg_alpha': None,  # L1 reg - faster under high dimensionality
# 'reg_lambda': None,  # L2 regularization - reduce overfitting
# 'scale_pos_weight': None,  # val > 0 to enable faster convergence under imbalanced class
# 'tree_method': None
# 'base_score': None,
# 'booster': None,
# 'colsample_bylevel': None,
# 'colsample_bynode': None,
# 'importance_type': 'gain',
# 'interaction_constraints': None,
# 'max_delta_step': None,
# 'monotone_constraints': None,
# 'n_jobs': None,
# 'num_parallel_tree': None,