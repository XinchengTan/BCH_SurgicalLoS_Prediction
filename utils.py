from collections import defaultdict, Counter

from IPython.display import display
from sklearn.utils import shuffle
from time import time
from tqdm import tqdm
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

import globals
from globals import *
from c1_data_preprocessing import Dataset


def make_k_all_feature_datasets(dashb_df: pd.DataFrame,
                                k=1,
                                outcome=globals.NNT,
                                onehot_cols=[],
                                discretize_cols=None,
                                remove_o2m=(True, True),
                                scaler='robust',
                                scale_numeric_only=True):
  code2decile_ftrs2aggf = {
    CCSR: {
      CCSR_DECILE: 'max', f'{CCSR}_COUNT': 'max',
      f'{CCSR}_MEDIAN': 'max', f'{CCSR}_SD': 'mean',
      f'{CCSR}_QT25': 'max', f'{CCSR}_QT75': 'max'
    },
    CPT: {
      CPT_DECILE: 'max', f'{CPT}_COUNT': 'max',
      f'{CPT}_MEDIAN': 'max', f'{CPT}_SD': 'mean',
      f'{CPT}_QT25': 'max', f'{CPT}_QT75': 'max'
    },
    PPROC: {
      PPROC_DECILE: 'max', f'{PPROC}_COUNT': 'max',
      f'{PPROC}_MEDIAN': 'max', f'{PPROC}_SD': 'mean',
      f'{PPROC}_QT25': 'max', f'{PPROC}_QT75': 'max'
    },
    MED3: {
      MED3_DECILE: 'max', f'{MED3}_COUNT': 'max',
      f'{MED3}_MEDIAN': 'max', f'{MED3}_SD': 'mean',
      f'{MED3}_QT25': 'max', f'{MED3}_QT75': 'max'
    },
  }
  datasets = []
  for i in range(k):
    df = dashb_df.sample(frac=1).reset_index(drop=True)
    dataset = Dataset(df, outcome, FEATURES_ALL_NO_WEIGHT, onehot_cols=onehot_cols,
                      discretize_cols=discretize_cols, col2decile_ftrs2aggf=code2decile_ftrs2aggf,
                      remove_o2m=remove_o2m, scaler=scaler, scale_numeric_only=scale_numeric_only)
    datasets.append(dataset)
  return datasets


def get_onehot_column_idxs(onehot_col, feature_names):
  oh_prefix = onehot_col + '_OHE' if onehot_col not in DRUG_COLS else 'MED%s_OHE_' % list(filter(str.isdigit, onehot_col))[0]
  oh_mask = np.char.startswith(feature_names, oh_prefix)
  return np.flatnonzero(oh_mask)


def exclude_onehot_cols(onehot_cols, feature_names, Xtrain, Xtest):
  full_oh_mask = np.zeros(len(feature_names)).astype(bool)
  feature_names = np.array(feature_names)
  for oh_col in onehot_cols:
    oh_prefix = oh_col + '_OHE' if oh_col not in DRUG_COLS else 'MED%s_OHE_' % list(filter(str.isdigit, oh_col))[0]
    oh_mask = np.char.startswith(feature_names, oh_prefix)
    full_oh_mask = np.ma.mask_or(full_oh_mask, oh_mask)
    print(oh_prefix, ': ', np.sum(oh_mask))
  keep_idxs = np.flatnonzero(~full_oh_mask)
  Xtrain, Xtest = Xtrain[:, keep_idxs], Xtest[:, keep_idxs]
  return Xtrain, Xtest


# Generate k dataset objects for k-fold cross validation
def gen_kfolds_datasets(df, kfold, features, shuffle_df=False, outcome=globals.NNT, onehot_cols=[],
                        discretize_cols=None, col2decile_ftr2aggf=None, cohort=globals.COHORT_ALL,
                        remove_o2m=(True, True), scaler='robust', scale_numeric_only=True):
  if shuffle_df:
    df = df.sample(frac=1).reset_index(drop=True)
  else:
    df = df.reset_index(drop=True)

  N, test_pct = df.shape[0], 1.0 / kfold
  datasets = []
  for k in tqdm(range(kfold)):
    test_idxs = np.arange(int(k * test_pct * N), int((k+1) * test_pct * N))
    dataset_k = Dataset(df, outcome, features, onehot_cols=onehot_cols, discretize_cols=discretize_cols,
                        col2decile_ftrs2aggf=col2decile_ftr2aggf, cohort=cohort, test_idxs=test_idxs,
                        remove_o2m=remove_o2m, scaler=scaler, scale_numeric_only=scale_numeric_only)
    datasets.append(dataset_k)

  return datasets


def gen_kfolds_Xytrain(Xtrain, ytrain, kfold, shuffle_data=True) -> List[Tuple]:
  if shuffle_data:
    Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=0)
  N = Xtrain.shape[0]
  val_pct = 1. / kfold
  kfolds_Xytrain = []
  sample_cnt = 0
  for k in range(kfold):
    if k < kfold - 1:
      val_idxs = np.arange(int(k * val_pct * N), int((k+1) * val_pct * N))
    else:
      val_idxs = np.arange(int(k * val_pct * N), N)
    kfolds_Xytrain.append((Xtrain[val_idxs, :], ytrain[val_idxs]))
    sample_cnt += len(val_idxs)
  assert N == sample_cnt, "%d samples are discarded" % (abs(N - sample_cnt))
  return kfolds_Xytrain


# Drop outliers (i.e. patient who has an extremely long LOS)
def drop_outliers(dashboard_df, exclude_outliers=0, inplace=True):
  if exclude_outliers > 0:
    print("Initial number of rows", dashboard_df.shape[0])
    if exclude_outliers == 0:
      return dashboard_df
    else:
      if inplace:
        dashboard_df.drop(index=dashboard_df.nlargest(exclude_outliers, 'LENGTH_OF_STAY').index, inplace=True)
        ret_df = dashboard_df
      else:
        ret_df = dashboard_df.drop(index=dashboard_df.nlargest(exclude_outliers, 'LENGTH_OF_STAY').index)
      print("After dropping outliers (number of rows)", ret_df.shape[0])
      return ret_df


# Generate the input dict to EnsembleModel
def gen_md2AllClfs(md2reg=None, md2multiclf=None, md2binclfs=None):
  md2allclfs = defaultdict(lambda : defaultdict(list))
  if md2reg is not None:
    for md, clf in md2reg.items():
      md2allclfs[md][globals.REG].append(clf)
  if md2multiclf is not None:
    for md, clf in md2multiclf.items():
      md2allclfs[md][globals.MULTI_CLF].append(clf)
  if md2binclfs is not None:
    for md, cutoff2clf in md2binclfs.items():
      md2allclfs[md][globals.BIN_CLF] = [cutoff2clf[c] for c in globals.NNT_CUTOFFS]
  return md2allclfs


# Outputs groups of rows that are identical in feature values, but different outcomes
def check_one_to_many_in_X_updated(dataset: Dataset, discretize_cols=None, check_train=True, verbose=True):
  """
  TODO: obesity definition according to CDC: https://www.cdc.gov/obesity/childhood/defining.html
  # weightz_bins = [float('-inf'), np.percentile(X[:, idx], 5), np.percentile(X[:, idx], 85),
  #                 np.percentile(X[:, idx], 95), float('inf')]
  """
  if check_train:
    X, y, feature_names = np.copy(dataset.Xtrain), np.copy(dataset.ytrain), dataset.feature_names
  else:
    X, y, feature_names = np.copy(dataset.Xtest), np.copy(dataset.ytest), dataset.feature_names
  N = np.shape(X)[0]
  print('Total number of cases: ', N)

  # Modify data matrix with discretized columns by request
  if discretize_cols:
    print('[utils] discretize cols not called')
    # discretize_columns(X, feature_names, discretize_cols, inplace=True)

  # Get duplicated feature rows with corresponding outcome
  start = time()
  Xdf = pd.DataFrame(X, columns=feature_names)
  dup_row_mask = Xdf.duplicated(keep=False)
  Xydf_dup = Xdf[dup_row_mask]
  y_dup = y[dup_row_mask]
  Xydf_dup['Outcome'] = y_dup
  print("[INFO] Get duplicated rows df took %.2f sec" % (time() - start))
  print("Number of rows with duplications: %d" % len(Xydf_dup))

  # Groupby all features, filter by group size (keep those > 1)
  start = time()
  Xydf_dup_group = Xydf_dup.groupby(by=feature_names)
  print("[INFO] Groupby all features took %.2f sec" % (time() - start))
  o2m_df = Xydf_dup_group.filter(lambda x: len(x.value_counts().index) > 1)
  pure_dup_df = Xydf_dup_group.filter(lambda x: len(x.value_counts().index) == 1)
  print("[INFO] Get pure duplicates and one-to-many cases took %.2f sec" % (time() - start))
  print("Number of one-to-many cases: ", len(o2m_df))
  print("Number of pure duplicates: ", len(pure_dup_df))

  # TODO: groupby sort by count
  if verbose:
    toTuple_ftrs = ['CPTS', 'ICD10S', 'CCSRS', 'CPT_GROUPS']
    sort_by_ftrs = ['PRIMARY_PROC', 'CPTS', 'ICD10S']
    grouby_ftrs = [f for f in globals.FEATURE_COLS_NO_OS if f != 'SURG_CASE_KEY']
    o2m_df_view = dataset.cohort_df[globals.FEATURE_COLS_NO_OS+['LENGTH_OF_STAY']].iloc[o2m_df.index].copy()
    o2m_df_view['LENGTH_OF_STAY'] = o2m_df_view['LENGTH_OF_STAY'].apply(lambda x: min(globals.MAX_NNT+1, int(x)))
    o2m_df_view[toTuple_ftrs] = o2m_df_view[toTuple_ftrs].applymap(lambda x: tuple(sorted(x)))
    o2m_df_view = o2m_df_view.groupby(by=grouby_ftrs).size().reset_index(name='Counts')  # TODO: why are there count == 1??? - age, weight

    pure_dup_df_view = dataset.cohort_df[globals.FEATURE_COLS_NO_OS+['LENGTH_OF_STAY']].iloc[pure_dup_df.index].copy()
    pure_dup_df_view['LENGTH_OF_STAY'] = pure_dup_df_view['LENGTH_OF_STAY'].apply(lambda x: min(globals.MAX_NNT+1, int(x)))
    pure_dup_df_view[toTuple_ftrs] = pure_dup_df_view[toTuple_ftrs].applymap(lambda x: tuple(sorted(x)))


    display(o2m_df_view.sort_values(by=sort_by_ftrs).head(30))
    display(pure_dup_df_view.sort_values(by=sort_by_ftrs).head(30))

  return o2m_df, pure_dup_df


# Outputs groups of rows that are identical in feature values, but different outcomes
def check_one_to_many_in_X(dataset: Dataset, discretize_cols=None, check_train=True, verbose=True):
  """
  TODO: obesity definition according to CDC: https://www.cdc.gov/obesity/childhood/defining.html
  # weightz_bins = [float('-inf'), np.percentile(X[:, idx], 5), np.percentile(X[:, idx], 85),
  #                 np.percentile(X[:, idx], 95), float('inf')]
  """
  if check_train:
    X, y, feature_names = np.copy(dataset.Xtrain), np.copy(dataset.ytrain), dataset.feature_names
  else:
    X, y, feature_names = np.copy(dataset.Xtest), np.copy(dataset.ytest), dataset.feature_names
  N = np.shape(X)[0]

  # Modify data matrix with discretized columns by request
  if discretize_cols:
    print('[utils] discretize col not called')
    # discretize_columns(X, feature_names, discretize_cols, inplace=True)

  start = time()
  # Track indices of feature-value duplicate rows
  identical_cases2idxs = defaultdict(set)
  for i in tqdm(range(N), "Building identical case to idx mapping"):
    row_bytes = X[i, :].tobytes()
    for j in range(i+1, N):
      if np.array_equal(X[i,:], X[j,:]):
        identical_cases2idxs[row_bytes] = identical_cases2idxs[row_bytes].union({i,j})
  print("[INFO]Get cases with duplications took %.2f sec\n" % (time() - start))

  start = time()
  # Filter cases to keep those with multiple outcome values
  one_to_many_cases, pure_duplicates = set(), set()
  identical_cases2outcomes = defaultdict(list)
  for row_bytes, idxs in tqdm(identical_cases2idxs.items(), "Filtering O2M & Duplicates"):
    identical_cases2outcomes[row_bytes] = [y[i] for i in idxs]
    if len(set(identical_cases2outcomes[row_bytes])) > 1:
      one_to_many_cases.add(row_bytes)
    else:
      pure_duplicates.add(row_bytes)
  print("[INFO]Get pure duplicates vs one-to-many cases took %.2f sec\n" % (time() - start))

  # Output basic analytical results
  print("Total number of cases: %d" % N)
  print("Number of One-to-many Occurrences: %d" % len(one_to_many_cases))
  o2m = sorted([(len(identical_cases2outcomes[k]), sorted(Counter(identical_cases2outcomes[k]).items()),
                 dataset.cohort_df.iloc[int(identical_cases2outcomes[k][0])][['PRIMARY_PROC', 'CCSRS', 'ICD10S']])
                for k in one_to_many_cases],
               key=lambda x: x[0], reverse=True)
  o2m_cnt, o2m_majority_cases_cnt = 0, 0
  for cnt, outcome_counter, comorbidity in o2m:
    o2m_cnt += cnt
    o2m_majority_cases_cnt += max([counts for _, counts in outcome_counter])
    if verbose:
      print("Count = {cnt}  {counter} \n"
            "- {pproc}, {ccsr}\n"
            "- {icd10}".format(cnt=cnt, counter=outcome_counter,pproc=comorbidity['PRIMARY_PROC'],
                               ccsr=set(comorbidity['CCSRS']), icd10=comorbidity['ICD10S']))
  print("\nTotal O2M affected cases: ", o2m_cnt)
  print("Naive Pick-majority Accuracy: %.2f%%" % (100 * (o2m_majority_cases_cnt + N - o2m_cnt) / N))

  print("\n\nNumber of Pure Duplicates: %d" % len(pure_duplicates))
  dups = sorted([(len(identical_cases2outcomes[k]),
                  dataset.cohort_df.iloc[int(identical_cases2outcomes[k][0])][['PRIMARY_PROC', 'CCSRS', 'ICD10S']])
                 for k in pure_duplicates], key=lambda x:x[0], reverse=True)
  dup_cnt = 0
  for cnt, comorbidity in dups:
    dup_cnt += cnt
    if verbose:
      print("Count = {cnt}\n"
            "- {pproc}, {ccsr}\n"
            "- {icd10}".format(cnt=cnt, pproc=comorbidity['PRIMARY_PROC'],
                               ccsr=set(comorbidity['CCSRS']), icd10=comorbidity['ICD10S']))
  print("Total Pure Duplicated cases: %d" % dup_cnt)
  return identical_cases2outcomes, one_to_many_cases, pure_duplicates, identical_cases2idxs

