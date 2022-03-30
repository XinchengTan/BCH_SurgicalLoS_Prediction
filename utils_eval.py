import numpy as np
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Sized

from c1_data_preprocessing import Dataset
from globals import *
import utils_plot


# Returns the classifier's name in a readable format
def get_clf_name(clf):
  try:
    md_name = clf.model_type
    assert clf.__class__.__name__ == 'SafeOneClassWrapper'
  except AttributeError:
    md_name = clf.__class__.__name__
  return md_name


# Initialize an empty pd.DataFrame with column names
def get_default_perf_df(scorers, outcome=NNT):
  columns = ['Trial', 'Xtype', 'Cohort', 'Model', 'Count', 'Year'] + scorers
  if outcome == NNT:
    return pd.DataFrame(columns=columns + [f'Count_class{c}' for c in NNT_CLASSES])
  else:
    return pd.DataFrame(columns=columns)  # todo: binary class fetch class label & append to 'columns'


# If certain evaluation metrics does not apply, display -1
def get_placeholder_perf_scores_dict(scorers: List) -> Dict:
  return {s: -1.0 for s in scorers}


# Returns the year label (either a single year, or a string of year range)
def get_year_label(years, dataset: Dataset = None) -> str:
  if years is None:
    return 'All' if dataset is None else f'{min(dataset.year_to_case_keys)} - {max(dataset.year_to_case_keys)}'
  elif len(years) > 1:
    return f'{min(years)} - {max(years)}'
  else:
    return str(years[0])


# Append a generic row (with any use-defined columns) to perf_df
def append_perf_row_generic(perf_df, score_dict: Dict, info_col_dict: Dict[str, Any]):
  score_dict.update(info_col_dict)
  perf_df = perf_df.append(score_dict, ignore_index=True)
  return perf_df


def append_perf_row(perf_df, trial, md, scores_row_dict: Dict):
  scores_row_dict['Model'] = md
  scores_row_dict['Trial'] = trial
  perf_df = perf_df.append(scores_row_dict, ignore_index=True)
  return perf_df


def append_perf_row_surg(surg_perf_df: pd.DataFrame, trial, scores_row_dict):
  if scores_row_dict is None or len(scores_row_dict) == 0:
    return surg_perf_df
  scores_row_dict['Trial'] = trial
  scores_row_dict['Model'] = 'Surgeon-train'
  surg_perf_df = surg_perf_df.append(scores_row_dict, ignore_index=True)
  return surg_perf_df


# Computes the count for each true class label
# TODO: update this for binary clf
def get_class_count(y, outcome=NNT):
  counter = defaultdict(int)
  if outcome == NNT:
    for cls in NNT_CLASSES:
      counter[f'Count_class{cls}'] = list(y).count(cls)
  elif outcome in BINARY_NNT_SET:
    counter[f'Count_[{outcome}]'] = sum(y)
    counter[f'Count_[{NNT} > {outcome[-1]}]'] = len(y) - sum(y)
  elif outcome in PHYSIO_DECLINE_SET:
    counter[f'Count_[{outcome}]'] = sum(y)
    counter[f'Count_[NO_DECLINE]'] = len(y) - sum(y)
  return counter


# Type-cast all count columns to numeric
def to_numeric_count_cols(perf_df: pd.DataFrame):
  for col in perf_df.columns:
    if col.startswith('Count'):
      perf_df[col] = pd.to_numeric(perf_df[col])
  return perf_df


# Format pd.DataFrame row-wise
def format_row_wise(styler, formatter):
  for row, row_formatter in formatter.items():
    if row in styler.index:
      row_num = styler.index.get_loc(row)

      for col_num in range(len(styler.columns)):
        styler._display_funcs[(row_num, col_num)] = row_formatter
  return styler


# Format numbers and floats in perf df (column-wise)
def format_perf_df(perf_df: pd.DataFrame):
  # define actual formatter to be applied on perf_df (metric name with suffixes)
  formatter_ret = deepcopy(SCR_FORMATTER)
  for scr in perf_df.columns.to_list():
    if scr.startswith('Count'):
      formatter_ret[scr] = '{:.0f}'.format
    elif scr not in {'Model', 'Xtype', 'Cohort', 'Trial', 'Year'}:
      if scr.startswith(SCR_RMSE):
        formatter_ret[scr] = SCR_FORMATTER[SCR_RMSE]
      elif scr.startswith(SCR_AUC):
        formatter_ret[scr] = SCR_FORMATTER[SCR_AUC]
      elif scr.startswith(SCR_F1_BINCLF):
        formatter_ret[scr] = SCR_FORMATTER[SCR_F1_BINCLF]
      else:
        formatter_ret[scr] = SCR_FORMATTER[scr]  # default formatter

  perf_styler = perf_df.style.format(formatter_ret)
  return perf_styler


# Display confusion matrix of one or more models on the dataset where it achieves its median performance across k trials
def show_confmat_of_median_perf_for_mds(perf_df, model_to_confmats, which_md, Xtype, criterion=SCR_ACC):
  which_md = str(which_md)
  if which_md.lower() == 'all':
    for md, kt_to_confmats in model_to_confmats.items():
      show_confmat_of_median_perf_(perf_df, kt_to_confmats, md, Xtype, criterion)
  else:
    kt = show_confmat_of_median_perf_(perf_df, model_to_confmats[which_md], which_md, Xtype, criterion)
    if 'Surgeon' in model_to_confmats:
      utils_plot.plot_confusion_matrix(model_to_confmats[SURGEON][kt], SURGEON, Xtype)


# Display the confusion matrix of a particular model on the dataset where it achieves its median perf across k trials
def show_confmat_of_median_perf_(perf_df, kt_to_confmats, md, Xtype, criterion):
  md_name = clf2name[md] if md != SURGEON else md
  criterion_sorted = perf_df[perf_df['Model'] == md_name].sort_values(by=criterion).reset_index(drop=True)
  kt = criterion_sorted.iloc[len(kt_to_confmats) // 2]['Trial']
  utils_plot.plot_confusion_matrix(kt_to_confmats[kt], md_name, Xtype)
  return kt