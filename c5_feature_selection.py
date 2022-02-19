import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, chi2, mutual_info_classif
from sklearn.utils import shuffle
from typing import List, Dict, Optional
from tqdm import tqdm

import utils
from globals import *
from c1_data_preprocessing import Dataset
from c4_model_perf import MyScorer
from jyp2_modeling import run_classifier

# [done] 0. Double check for nans in miles traveld, state code
# TODO 1. Finish stepwise addition func & Generate result tables & performance
# [done] 2. Add CCSR deciles & Test
# TODO 3. Add combined medication level based decile features (fillna with the previous level, e.g. level1_medx_generic)
# TODO 4. Add violin plot across **pproc**, cpt, ccsrs
# TODO 5. Think about how to handle one-to-many cases in the stepwise addition process


class FeatureSelector(object):

  def __init__(self):
    pass

  @staticmethod
  def selectKBest_fs(dataset: Dataset, n_features_to_select, score_func, cv=5):
    assert score_func in {chi2, mutual_info_classif}
    feature_to_select_times = np.zeros(len(dataset.feature_names))
    X, y = np.copy(dataset.Xtrain), np.copy(dataset.ytrain)
    for i in range(cv):
      X, y = shuffle(X, y, random_state=0)
      skb = SelectKBest(score_func=score_func, k=n_features_to_select)
      skb.fit(X, y)
      feature_to_select_times += skb.get_support().astype(int)

    # Return the top K features
    selected_feature_idxs = (-feature_to_select_times).argsort()[:n_features_to_select]  # descending order, pick topK frequent ones
    return selected_feature_idxs, np.array(dataset.feature_names)[selected_feature_idxs]

  @ staticmethod
  def sequential_fs_sklearn(estimator,
                            dataset: Dataset,
                            n_features_to_select,
                            direction='forward',
                            scoring='accuracy',
                            cv=5):
    # Xdata, ydata should be the training data (with validation proportion) for modeling
    # TODO: potential data loss/bias given that it can't handle NAs by feature set (cases are excluded altogether)
    sfs = SequentialFeatureSelector(estimator, n_features_to_select=n_features_to_select, direction=direction,
                                    scoring=scoring, cv=cv, n_jobs=-1)
    sfs.fit(dataset.Xtrain, dataset.ytrain)
    selected_features = np.array(dataset.feature_names)[sfs.get_support()]
    print("\nSelected features: ", selected_features)
    return sfs  # sfs.get_support(), selected_features,


  # TODO: following methods
  @staticmethod
  def stepwise_batch_addition_fs(estimator,
                                 ktrial_datasets: List[Dataset],
                                 base_features=None,
                                 add_features=None,
                                 scorers=None,
                                 rand_state=SEED):
    if base_features is None:
      base_features = [PPROC_DECILE]
    if add_features is None:
      add_features = [
                       CPT_DECILE, MED3_DECILE,
                       f'{PPROC}_COUNT', f'{CPT}_COUNT', f'{MED3}_COUNT',
                       f'{PPROC}_MEDIAN', f'{CPT}_MEDIAN', f'{MED3}_MEDIAN',
                       f'{PPROC}_QT25', f'{CPT}_QT25', f'{MED3}_QT25',
                       f'{PPROC}_QT75', f'{CPT}_QT75', f'{MED3}_QT75',
                       f'{PPROC}_SD', f'{CPT}_SD', f'{MED3}_SD',
                       AGE, GENDER, MILES, STATE, REGION, LANGUAGE, INTERPRETER, PROBLEM_CNT,
                     ] + OS_CODE_LIST + [PRIMARY_PROC, CPTS, CCSRS, DRUG_COLS[2]]
    if scorers is None:
      scorers = MyScorer.get_scorer_dict([SCR_ACC, SCR_ACC_BAL, SCR_ACC_ERR1, SCR_ACC_ERR2, SCR_OVERPRED, SCR_UNDERPRED,
                                          SCR_RMSE])
    train_perf_df = pd.DataFrame(columns=['Trial', 'nth_feature'] + list(scorers.keys()))
    test_perf_df = pd.DataFrame(columns=['Trial', 'nth_feature'] + list(scorers.keys()))
    for trial_i, dataset in tqdm(enumerate(ktrial_datasets), 'trial'):
      print('Trial %d' % trial_i)
      train_perf_df, test_perf_df = stepwise_batch_addition_single_trial(
        trial_i, estimator, dataset, base_features=base_features, features_to_add=add_features,
        scorers=scorers, train_perf_df=train_perf_df, test_perf_df=test_perf_df, rand_state=rand_state)

    return train_perf_df, test_perf_df

  @staticmethod
  def simulated_annealing_fs():
    pass

  @staticmethod
  def resampled_simulated_annealing_fs():
    pass


# Apply feature selection technique and evaluate on test data
def feature_selection_with_eval(estimator, ktrial_datasets, scorers, fs_scorer, n_feature_values, fs_how='sfs'):
  # kwargs: scaler, remove_o2m, discretize, outcome
  if scorers is None:
    scorers = MyScorer.get_scorer_dict([SCR_ACC, SCR_ACC_BAL, SCR_ACC_ERR1, SCR_ACC_ERR2, SCR_OVERPRED, SCR_UNDERPRED,
                                        SCR_RMSE])
  train_perf_df = pd.DataFrame(columns=['Trial', 'n_features'] + list(scorers.keys()))
  test_perf_df = pd.DataFrame(columns=['Trial', 'n_features'] + list(scorers.keys()))

  feature_select_counts = np.zeros(len(ktrial_datasets[0].feature_names))
  # save selected features somewhere!!
  for trial, dataset in tqdm(enumerate(ktrial_datasets), 'sfs-Trial'):
    if fs_how == 'sfs':
      for n_ftrs in n_feature_values:
        print('\nTrial %d, num_features = %d' % (trial, n_ftrs))
        # With cross validation, sequentially select n_ftrs
        sfs = FeatureSelector.sequential_fs_sklearn(estimator, dataset, n_features_to_select=n_ftrs,
                                                    scoring=fs_scorer, cv=5)
        # Fit estimator on Xtrain
        Xtrain, Xtest = sfs.transform(dataset.Xtrain), sfs.transform(dataset.Xtest)
        estimator.fit(Xtrain, dataset.ytrain)
        # Evaluate on Xtrain and Xtest, using only the selected features
        train_pred, test_pred = estimator.predict(Xtrain), estimator.predict(Xtest)
        train_score_dict = MyScorer.apply_scorers(scorers.keys(), dataset.ytrain, train_pred)
        test_score_dict = MyScorer.apply_scorers(scorers.keys(), dataset.ytest, test_pred)
        print(test_score_dict)
        # Save the results to a dataframe
        train_score_dict['Trial'], test_score_dict['Trial'] = trial, trial
        train_score_dict['n_features'], test_score_dict['n_features'] = n_ftrs, n_ftrs
        train_perf_df.append(train_score_dict, ignore_index=True)
        test_perf_df.append(test_score_dict, ignore_index=True)
        # Increment counter for selected features
        feature_select_counts[sfs.get_support(indices=True)] += 1
    else:
      raise NotImplementedError

  return train_perf_df, test_perf_df, {ktrial_datasets[0].feature_names[i]: feature_select_counts[i] for i in range(len(feature_select_counts))}


# TODO: add visualization (1. Line plot of num_features ~ Test perfs -- diff color for diff models,
#                          2. Heatmap of selected feature count)



def stepwise_batch_addition_single_trial(trial_i: int,
                                         estimator,
                                         dataset: Dataset,
                                         base_features=None,
                                         features_to_add=None,
                                         scorers=None,
                                         train_perf_df=None,
                                         test_perf_df=None,
                                         rand_state=SEED):
  """
  This function iteratively adds features, one type at a time, to a pre-defined classifier trained on dataset.Xtrain
  and evaluated on dataset.Xtest.

  Each feature batch corresponds to a raw column in a prepared dataframe, except for all decile-related features.

  :param estimator: a classifier (sklearn Model object)
  :param dataset: a Dataset object with full feature set
  :param base_features: starting feature set
  :param features_to_add: features to add, must not overlap with 'base_features'
  :param rand_state: seed
  :return:
  """
  full_features = np.array(dataset.feature_names)
  cur_feature, ftr_in_use_idxs = base_features[0], np.where(np.in1d(full_features, base_features))[0]
  if train_perf_df is None:
    train_perf_df = pd.DataFrame(columns=['Trial', 'nth_feature'] + list(scorers.keys()))
  if test_perf_df is None:
    test_perf_df = pd.DataFrame(columns=['Trial', 'nth_feature'] + list(scorers.keys()))
  for new_feature in tqdm(features_to_add, f'Trial {trial_i} - features'):
    # Train on features in use
    Xtrain, Xtest = dataset.Xtrain[:, ftr_in_use_idxs], dataset.Xtest[:, ftr_in_use_idxs]
    estimator.fit(Xtrain, dataset.ytrain)

    # Evaluate on Xtrain and Xtest, using only the selected features
    train_pred, test_pred = estimator.predict(Xtrain), estimator.predict(Xtest)
    train_score_dict = MyScorer.apply_scorers(scorers.keys(), dataset.ytrain, train_pred)
    test_score_dict = MyScorer.apply_scorers(scorers.keys(), dataset.ytest, test_pred)

    # Save the results to a dataframe
    train_score_dict['Trial'], test_score_dict['Trial'] = trial_i, trial_i
    train_score_dict['nth_feature'], test_score_dict['nth_feature'] = cur_feature, cur_feature
    train_perf_df = train_perf_df.append(train_score_dict, ignore_index=True)
    test_perf_df = test_perf_df.append(test_score_dict, ignore_index=True)

    # Add a new feature
    if new_feature in {STATE, REGION, LANGUAGE}:
      ftr_col_idxs = np.where(np.in1d(full_features, COL2DUMMIES[new_feature]))[0]
    elif new_feature in {PRIMARY_PROC, CPTS, CCSRS, DRUG_COLS[2]}:
      ftr_col_idxs = utils.get_onehot_column_idxs(new_feature, full_features)
    else:
      ftr_col_idxs = np.where(np.in1d(full_features, new_feature))[0]
    ftr_in_use_idxs = np.concatenate([ftr_in_use_idxs, ftr_col_idxs])
    cur_feature = new_feature

  return train_perf_df, test_perf_df



  # TODO: What to do with removing o2m along the step-wise addition???

  # 0. start with PProc decile as a base feature
  # 1. add other deciles one by one (cpt, med3, ccsr)
  # - 1.1 add code count
  # - 1.2 add qt25
  # - 1.3 add qt75
  # - 1.4 add code SD
  # 2. add age
  # 3. add gender
  # 4. add major region
  # 5. add state code
  # 6. add miles
  # 7. add language
  # 8. add interpreter needed
  # 9. add OS code
  # 10. add PProc indicators
  # 11. add CPT indicators
  # 12. add CCSR indicators
  # 13. add Med3 indicators

  # For each iteration, try 1. no input scaling  2. scale numeric features only  3. scale all -- RobustScaler
