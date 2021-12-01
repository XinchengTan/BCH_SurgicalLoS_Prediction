from collections import defaultdict, Counter

from tqdm import tqdm
import numpy as np
import pandas as pd
from . import globals
from .c1_data_preprocessing import Dataset


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
def check_one_to_many_in_X(dataset: Dataset, discretize_cols=None, check_train=True):
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
  for dis_col in discretize_cols:
    idx = feature_names.index(dis_col)
    if dis_col == 'AGE_AT_PROC_YRS':
      X[:, idx] = np.digitize(X[:, idx], globals.AGE_BINS)
    elif dis_col == 'WEIGHT_ZSCORE':
      weightz_bins = [float('-inf'), -4, -2, -1, 1, 2, 4, float('inf')]
      print("Weight z-score bins: ", weightz_bins)
      X[:, idx] = np.digitize(X[:, idx], weightz_bins)
    else:
      raise Warning("%s discretization is not available yet!" % dis_col)

  # Track indices of feature-value duplicate rows
  identical_cases2idxs = defaultdict(set)
  for i in tqdm(range(N), "Building identical case mapping"):
    row_bytes = X[i, :].tobytes()
    for j in range(i+1, N):
      if np.array_equal(X[i,:], X[j,:]):
        identical_cases2idxs[row_bytes] = identical_cases2idxs[row_bytes].union({i,j})

  # Filter cases to keep those with multiple outcome values
  one_to_many_cases, pure_duplicates = set(), set()
  identical_cases2outcomes = defaultdict(list)
  for row_bytes, idxs in tqdm(identical_cases2idxs.items(), "Filtering O2M & Duplicates"):
    identical_cases2outcomes[row_bytes] = [y[i] for i in idxs]
    if len(set(identical_cases2outcomes[row_bytes])) > 1:
      one_to_many_cases.add(row_bytes)
    else:
      pure_duplicates.add(row_bytes)

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
    print("Count = {cnt}\n"
          "- {pproc}, {ccsr}\n"
          "- {icd10}".format(cnt=cnt, pproc=comorbidity['PRIMARY_PROC'],
                             ccsr=set(comorbidity['CCSRS']), icd10=comorbidity['ICD10S']))
  print("Total Pure Duplicated cases: %d" % dup_cnt)
  return identical_cases2outcomes, one_to_many_cases, pure_duplicates, identical_cases2idxs
