import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, chi2, mutual_info_classif
from sklearn.utils import shuffle
from typing import List, Dict, Optional

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
  def selectKBest_fs(dataset: Dataset, n_features_to_select, score_func, cv=10):
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
                            cv=10):
    # Xdata, ydata should be the training data (with validation proportion) for modeling
    # TODO: potential data loss/bias given that it can't handle NAs by feature set (cases are excluded altogether)
    sfs = SequentialFeatureSelector(estimator, n_features_to_select=n_features_to_select, direction=direction,
                                    scoring=scoring, cv=cv, n_jobs=-1)
    sfs.fit(dataset.Xtrain, dataset.ytrain)
    selected_features = np.array(dataset.feature_names)[sfs.get_support()]
    print("\nSelected features: ", selected_features)

    # Compute train & test accuracy on selected features
    Xtrain, Xtest = sfs.transform(dataset.Xtrain), sfs.transform(dataset.Xtest)
    estimator.fit(Xtrain, dataset.ytrain)
    train_pred, test_pred = estimator.predict(Xtrain), estimator.predict(Xtest)
    return sfs.get_support(), selected_features, sfs, estimator

  @staticmethod
  def stepwise_batch_addition_fs(estimator,
                                 dashb_df: pd.DataFrame,
                                 ktrials: int,
                                 base_features=None,
                                 add_features=None,
                                 rand_state=SEED):

    pass

  @staticmethod
  def simulated_annealing_fs():
    pass

  @staticmethod
  def resampled_simulated_annealing_fs():
    pass



def stepwise_addition_single_trial(clf,
                                   dashb_df: pd.DataFrame,
                                   base_features: Optional[List[str]],
                                   features_to_add: Optional[List[str]],
                                   rand_state=SEED):
  """
  This function iteratively adds features, one at a time, to a pre-defined classifier trained on dataset.Xtrain
  and evaluated on dataset.Xtest.

  :param clf: a classifier (sklearn Model object)
  :param dashb_df: a prepared dataframe
  :param base_features: starting feature set
  :param features_to_add: features to add, must not overlap with 'base_features'
  :param rand_state: seed
  :return:
  """
  if base_features is None:
    base_features = [PPROC_DECILE]
  if features_to_add is None:
    features_to_add = [
      CPT_DECILE, MED3_DECILE,
      f'{PPROC}_COUNT', f'{CPT}_COUNT', f'{MED3}_COUNT',
      f'{PPROC}_MEDIAN', f'{CPT}_MEDIAN', f'{MED3}_MEDIAN',
      f'{PPROC}_QT25', f'{CPT}_QT25', f'{MED3}_QT25',
      f'{PPROC}_QT75', f'{CPT}_QT75', f'{MED3}_QT75',
      f'{PPROC}_SD', f'{CPT}_SD', f'{MED3}_SD',
      AGE, GENDER, REGION, STATE, MILES, LANGUAGE, INTERPRETER, PROBLEM_CNT,
      OS_CODES, PRIMARY_PROC, CPTS, CCSRS, DRUG_COLS[2]
    ]




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

  # TODO: What to do with removing o2m along the step-wise addition???

  return
