import numpy as np

from globals import *
from c1_data_preprocessing import Dataset
from c2_models_nnt import *
from c2_models_chews import *
from c4_model_perf import MyScorer


# -------------------------------------- Train models on k Datasets (Multi-class) --------------------------------------
def train_model_all_ktrials(models, k_datasets: Dict[Any, Dataset], cls_weight,
                            train_sda_only=False, train_surg_only=False) -> Dict:
  models = [LGR, KNN, RMFCLF, XGBCLF] if models is None else models  # GBCLF,
  k_model_dict = {}
  for k, dataset_k in tqdm(k_datasets.items()):
    # Fit models
    Xtrain, ytrain = dataset_k.get_Xytrain_by_case_key(dataset_k.train_case_keys,
                                                       sda_only=train_sda_only, surg_only=train_surg_only)
    model_dict = {}
    for md in models:
      print('md=', md)
      if dataset_k.outcome == NNT:
        clf = get_model(md, cls_weight=cls_weight)
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
                                   train_sda_only=False, train_surg_only=False):
  bin_k_model_dict = defaultdict(dict)
  for bin_nnt, kt_datasets in tqdm(bin_kt_datasets.items(), desc='Binary Outcomes'):
    print('Outcome: ', bin_nnt)
    bin_k_model_dict[bin_nnt] = train_model_all_ktrials(models, k_datasets=kt_datasets, cls_weight=cls_weight,
                                                        train_sda_only=train_sda_only, train_surg_only=train_surg_only)
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
def tune_model_randomSearch(md, X, y, kfold, scorers, n_iters=10, refit=True, calibrate=False):
  binary_cls = len(set(y)) < 3
  clf, param_space = gen_model_param_space(md, X, y, scorers=scorers, kfold=kfold)

  refit = get_refit_val(scorers, refit)
  rand_search = RandomizedSearchCV(clf, param_space, n_jobs=-1, refit=refit, cv=kfold, n_iter=n_iters,
                                   scoring=MyScorer.get_scorer_dict(scorers, binary_cls=binary_cls))
  rand_search.fit(X, y)
  return rand_search



def tune_model_gridSearch(md, X, y, scorers, kfold=5, refit=True):
  clf, param_space = gen_model_param_space(md, X, y, scorers, kfold)

  # Use GridSearchCV for hyperparameter tuning
  grid_search = GridSearchCV(estimator=clf, param_grid=param_space, scoring=scorers, cv=kfold, n_jobs=-1,
                             refit=refit, return_train_score=True, verbose=0)
  grid_search.fit(X, y)
  return


# ------------------------------------- Hyperparameter Tuning with GridSearchCV -------------------------------------
# TODO: add arg for outcome
def gen_model_param_space(md, X, y, scorers, kfold=5):
  n_frts = X.shape[1]
  minority_size = min(Counter(y).values())
  print("Minority-class size: ", minority_size)

  min_samples_split_max = int(minority_size * (1 - 1 / kfold))
  if md == LGR:
    param_space = {'Cs': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                   'l1_ratio': [0, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99, 1],
                   'class_weight': [None, 'balanced']}
    clf = LogisticRegression(random_state=SEED, penalty='elasticnet', solver='saga', max_iter=500)
  elif md == SVCLF:
    clf = SVC(random_state=SEED, probability=False)
    param_space = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                   'gamma': sorted(list({1 / n_frts, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001})),
                   'kernel': ['rbf'],
                   'class_weight': [None, 'balanced']
                   }
  elif md == SVC_POLY:
    clf = SVC(random_state=SEED)
    param_space = {'C': [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                   'kernel': ['poly'],
                   'degree': [2, 3, 4, 5],
                   'class_weight': [None, 'balanced']
                   }
  elif md == KNN:
    clf = KNeighborsClassifier()
    param_space = {
      'algorithm': ['ball_tree', 'kd_tree', 'brute'],
      'leaf_size': list(range(20, minority_size, 10)),
      'metric': ['minkowski'],
      'n_neighbors': list(range(5, minority_size + 1, minority_size // 10 + 1)),
      'p': [1, 2, 3],
      'weights': ['uniform', 'distance']
    }
  elif md == RMFCLF:
    clf = RandomForestClassifier(random_state=SEED)
    param_space = {
      'class_weight': [None, 'balanced'],
      'max_depth': [None] + list(range(2, 21, 2)),
      'max_features': sorted(set(list(range(2, n_frts // 2 + 1, n_frts // 20 + 1)) + [int(np.sqrt(n_frts))])),
      'max_leaf_nodes': [None] + list(range(5, 101, 5)),
      'max_samples': np.arange(0.1, 1, 0.1),
      'min_samples_leaf': [1] + [i for i in range(2, 17, 2)],
      'min_samples_split': list(range(2, min_samples_split_max, 3)),
      'n_estimators': [20, 30, 40, 50, 80, 100, 200, 300, 400]
    }
    if not param_space['min_samples_split']:
      raise ValueError("min_samples_split cannot < 2!")
  elif md == XGBCLF:
    # TODO: update eval_metric to 'auc' for binary clf, and 'rmse' for multi-class clf?
    clf = XGBClassifier(random_state=SEED)
    if n_frts > 500:
      colsample_bytree_range = np.arange(0.3, 1.01, 0.1)
    else:
      colsample_bytree_range = np.arange(0.8, 1.01, 0.05)
    scale_pos_weight_list = None
    # TODO: update scale_pos_weight param space, depending on outcome
    param_space = {
      'n_estimators': np.arange(50, 301, 50),
      'learning_rate': [0.01, 0.03, 0.1, 0.2, 0.3, 1],
      'subsample': np.arange(0.8, 1.01, 0.05),
      'colsample_bytree': colsample_bytree_range,
      'max_depth': [3, 4, 5, 6, 7],
      'gamma': [0, 0.1, 0.3, 1, 3, 5],
      'min_child_weight': [0.1, 0.3, 0.6, 1, 2],
      'scale_pos_weight': scale_pos_weight_list,
      'eval_metric': 'mlogloss',
    }
  elif md == GBCLF:
    # TODO: what's validation fraction?
    clf = GradientBoostingClassifier(random_state=SEED, validation_fraction=0.15, n_iter_no_change=3)
    param_space = {
      'class_weight': [None, 'balanced'],
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
      'class_weight': [None, 'balanced'],
      'max_depth': [None] + list(range(2, 21, 2)),
      'max_features': list(range(2, 1 + n_frts // 2, 10)) + [n_frts],
      'max_leaf_nodes': [None] + list(range(5, 101, 5)),
      'max_samples': np.arange(0.1, 1, 0.1),
      'min_samples_leaf': [1] + [i for i in range(2, 17, 2)],
      'min_samples_split': list(range(2, min_samples_split_max, 3)),
      'splitter': ['best', 'random'],
    }
  else:
    raise NotImplementedError("Model %s is not supported!" % md)

  return clf, param_space


## XGB other params:
# 'reg_alpha': None,  # L1 reg - faster under high dimensionality
# 'reg_lambda': None,  # L2 regularization - reduce overfitting
# 'scale_pos_weight': None,  # val > 0 to enable faster convergence under imbalanced class
# 'tree_method': None
# 'use_label_encoder': True,
# 'base_score': None,
# 'booster': None,
# 'colsample_bylevel': None,
# 'colsample_bynode': None,
# 'gpu_id': None,
# 'importance_type': 'gain',
# 'interaction_constraints': None,
# 'max_delta_step': None,
# 'monotone_constraints': None,
# 'n_jobs': None,
# 'num_parallel_tree': None,