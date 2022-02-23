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
                       CPT_DECILE, MED123_DECILE, CCSR_DECILE,
                       f'{PPROC}_COUNT', f'{CPT}_COUNT', f'{MED123}_COUNT', f'{CCSR}_COUNT',
                       f'{PPROC}_MEDIAN', f'{CPT}_MEDIAN', f'{MED123}_MEDIAN', f'{CCSR}_MEDIAN',
                       f'{PPROC}_QT25', f'{CPT}_QT25', f'{MED123}_QT25', f'{CCSR}_QT25',
                       f'{PPROC}_QT75', f'{CPT}_QT75', f'{MED123}_QT75', f'{CCSR}_QT75',
                       f'{PPROC}_SD', f'{CPT}_SD', f'{MED123}_SD', f'{CCSR}_SD',
                       AGE, GENDER, MILES, STATE, REGION, LANGUAGE, INTERPRETER, PROBLEM_CNT,
                     ] + OS_CODE_LIST + [PRIMARY_PROC, CPTS, CCSRS, DRUG_COLS[3]]
    if scorers is None:
      scorers = MyScorer.get_scorer_dict([SCR_ACC, SCR_ACC_BAL, SCR_ACC_ERR1, SCR_ACC_ERR2, SCR_OVERPRED, SCR_UNDERPRED,
                                          SCR_RMSE])
    train_perf_df = pd.DataFrame(columns=['Trial', 'nth_feature'] + list(scorers.keys()))
    test_perf_df = pd.DataFrame(columns=['Trial', 'nth_feature'] + list(scorers.keys()))
    for trial_i, dataset in tqdm(enumerate(ktrial_datasets), 'trial'):
      print('Trial %d' % trial_i)
      train_perf_df, test_perf_df = FeatureSelector._stepwise_batch_addition_single_trial(
        trial_i, estimator, dataset, base_features=base_features, features_to_add=add_features,
        scorers=scorers, train_perf_df=train_perf_df, test_perf_df=test_perf_df, rand_state=rand_state)

    return train_perf_df, test_perf_df

  @staticmethod
  def simulated_annealing_fs():
    pass

  @staticmethod
  def resampled_simulated_annealing_fs():
    pass

  @staticmethod
  def _stepwise_batch_addition_single_trial(trial_i: int,
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
    if train_perf_df is None:
      train_perf_df = pd.DataFrame(columns=['Trial', 'nth_feature'] + list(scorers.keys()))
    if test_perf_df is None:
      test_perf_df = pd.DataFrame(columns=['Trial', 'nth_feature'] + list(scorers.keys()))
    # Train and get performance on base features
    ftr_in_use_idxs = np.where(np.in1d(full_features, base_features))[0]
    estimator, train_perf_df, test_perf_df = train_and_get_perf_df(
      trial_i, base_features[-1], estimator, dataset.Xtrain[:, ftr_in_use_idxs], dataset.ytrain,
      dataset.Xtest[:, ftr_in_use_idxs], dataset.ytest, scorers, train_perf_df, test_perf_df
    )

    # todo: fix me and remove debugging code
    idxs = list(range(60)) + list(range(3750, len(full_features)))
    for i in idxs:
      print('i=', i, full_features[i])
    print('\nInitial features in use', ftr_in_use_idxs)
    print('\nFull XTrain shape', dataset.Xtrain.shape, 'Full Xtest shape', dataset.Xtest.shape)
    cum_ftr_cnt = len(ftr_in_use_idxs)

    # Add one feature (batch) at a time to base_features
    for nth_feature in tqdm(features_to_add, f'Trial {trial_i} - features'):
      # Get feature indices of 'nth_feature'
      if nth_feature in {STATE, REGION, LANGUAGE}:
        ftr_col_idxs = np.where(np.in1d(full_features, COL2DUMMIES[nth_feature]))[0]
      elif nth_feature in {PRIMARY_PROC, CPTS, CCSRS, DRUG_COLS[2], DRUG_COLS[3]}:
        ftr_col_idxs = utils.get_onehot_column_idxs(nth_feature, full_features)
      else:
        ftr_col_idxs = np.where(np.in1d(full_features, nth_feature))[0]
      print('\n', nth_feature, ": ", ftr_col_idxs)
      cum_ftr_cnt += len(ftr_col_idxs)
      ftr_in_use_idxs = np.concatenate([ftr_in_use_idxs, ftr_col_idxs])

      # Get Xtrain based on selected features
      Xtrain, Xtest = dataset.Xtrain[:, ftr_in_use_idxs], dataset.Xtest[:, ftr_in_use_idxs]
      print('Xtrain shape:', Xtrain.shape, 'Xtest shape:', Xtest.shape)

      # Train on Xtrain and evaluate on Xtrain & Xtest, and save perf to df
      estimator, train_perf_df, test_perf_df = train_and_get_perf_df(
        trial_i, nth_feature, estimator, Xtrain, dataset.ytrain, Xtest, dataset.ytest, scorers, train_perf_df, test_perf_df)

    print('Total utilized features: ', cum_ftr_cnt)
    print('Total available features: ', len(full_features))
    return train_perf_df, test_perf_df


def train_and_get_perf_df(trial_i, nth_ftr, estimator, Xtrain, ytrain, Xtest, ytest, scorers, train_perf_df, test_perf_df):
  # Train estimator on Xtrain
  estimator.fit(Xtrain, ytrain)

  # Evaluate on Xtrain and Xtest, using only the selected features
  train_pred, test_pred = estimator.predict(Xtrain), estimator.predict(Xtest)
  train_score_dict = MyScorer.apply_scorers(scorers.keys(), ytrain, train_pred)
  test_score_dict = MyScorer.apply_scorers(scorers.keys(), ytest, test_pred)

  # Save the results to a dataframe
  train_score_dict['Trial'], test_score_dict['Trial'] = trial_i, trial_i
  train_score_dict['nth_feature'], test_score_dict['nth_feature'] = nth_ftr, nth_ftr
  train_perf_df = train_perf_df.append(train_score_dict, ignore_index=True)
  test_perf_df = test_perf_df.append(test_score_dict, ignore_index=True)
  return estimator, train_perf_df, test_perf_df


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
