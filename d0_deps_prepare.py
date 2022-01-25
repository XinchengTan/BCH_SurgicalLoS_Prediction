# This script generates the following dependency files for the LoS prediction pipeline:
#   1. A list of cases that have one-to-many outcomes
#   2. A list of code indicators, ordered as they are in model training
#   3.

# TODO: pandas - sort columns

import numpy as np
import pandas as pd
import pickle
import warnings

from . import c1_data_preprocessing as dpp


# Generate a list of (unique) cases that have at least one identical case with a different outcome
def gen_o2m_cases(X, y, features, unique=True, save_fp=None, X_case_keys=None):
  Xydf = pd.DataFrame(X, columns=features)
  Xydf['Outcome'] = y
  if X_case_keys is not None:
    Xydf.insert(loc=0, column='SURG_CASE_KEY', value=X_case_keys)
  dup_mask = Xydf.duplicated(subset=features, keep=False)
  Xydf_dup = Xydf[dup_mask]

  # Get cases with identical features but different outcomes
  o2m_df = Xydf_dup\
    .groupby(by=features)\
    .filter(lambda x: len(x.value_counts().index) > 1)
  print("Covered %d o2m cases" % o2m_df.shape[0])
  if unique:
    o2m_df = o2m_df.drop_duplicates(subset=features)
    print("Contains %d unique o2m cases" % o2m_df.shape[0])
  o2m_df.drop(columns=['Outcome'], inplace=True)

  if save_fp:
    o2m_df.to_csv(save_fp, index=False)
  return o2m_df


# Remove the cases (one-to-many) in skip_cases_fp from a new dataset (X, y)
def discard_o2m_cases_from_historical_data_training(X, y, skip_cases_fp):
  # skip_cases_fp: file path of the historical cases that have multiple possible outcomes
  skip_cases_df = pd.read_csv(skip_cases_fp, index_col=False)
  assert skip_cases_df.shape[1] == X.shape[1], "Input data matrix must match with training data in features!"

  Xydf = pd.DataFrame(X, columns=skip_cases_df.columns)
  Xydf['Outcome'] = y
  Xydf = pd.merge(Xydf, skip_cases_df, how='left', on=skip_cases_df.columns.to_list(), indicator=True)
  Xydf = Xydf[Xydf['_merge'] == 'left_only']  # Left join minus inner join with skip_cases_df

  return Xydf[skip_cases_df.columns].to_numpy(), Xydf['Outcome'].to_numpy()


# Remove one-to-many cases from dataset (X, y)
def discard_o2m_cases(X, y, features):
  o2m_df = gen_o2m_cases(X, y, features)
  o2m_idx = o2m_df.index.to_list()
  print("One-to-many cases index: ", o2m_idx)
  print("#O2M cases: ", len(o2m_idx))
  idxs = np.delete(np.arange(X.shape[0]), o2m_idx)
  X, y = X[idxs, :], y[idxs]
  return X, y, idxs



# Generate a list of column names
def gen_feature_list(dataset: dpp.Dataset, save_fp=None):
  ftrs = dataset.feature_names
  if save_fp:
    np.savetxt(save_fp, ftrs, delimiter=', ', fmt='%s')
  return ftrs


def get_pproc_with_colidx(columns):
  return get_ftr_with_colidx(columns, prefix='PRIMARY_PROC')


def get_cptgrp_with_colidx(columns):
  return get_ftr_with_colidx(columns, prefix='CPT_GROUP')


def get_cpt_with_colidx(columns):
  return get_ftr_with_colidx(columns, prefix='CPT')


def get_ccsr_with_colidx(columns):
  return get_ftr_with_colidx(columns, prefix='CCSR')


def get_ftr_with_colidx(columns, prefix=None):
  if prefix is None:
    warnings.warn("Feature prefix is None, returning None")
    return None

  start_idx, end_idx = None, None
  for i in range(len(columns)):
    if columns[i].startswith(prefix):
      start_idx = i
      break

  for i in range(len(columns), -1, -1):
    if columns[i].startswith(prefix):
      end_idx = i
      break

  return columns[start_idx:end_idx+1], list(range(start_idx, end_idx+1))




