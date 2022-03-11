import pandas as pd
import numpy as np
from typing import Any, Dict, List

from copy import deepcopy
from globals import *
from c1_data_preprocessing import Dataset
from c4_model_perf import MyScorer, append_perf_row_generic, format_perf_df, get_clf_name, get_default_perf_df


def get_year_label(years):
  if years is None:
    return f'All'
  elif len(years) > 1:
    return f'{min(years)} - {max(years)}'
  else:
    return str(years[0])
  

def eval_cohort_clf(cohort_to_dataset: Dict[str, Dataset], cohort_to_clf: Dict[str, Any], scorers: List = None,
                    trial_i=None, years=None, train_perf_df=None, test_perf_df=None):
  if scorers is None:
    scorers = deepcopy(DEFAULT_SCORERS)
  outcome_type = list(cohort_to_dataset.values())[0].outcome
  if train_perf_df is None:
    train_perf_df = get_default_perf_df(scorers, outcome_type)
  if test_perf_df is None:
    test_perf_df = get_default_perf_df(scorers, outcome_type)

  year_label = get_year_label(years)

  # For each cohort dataset, evaluate the performance of the model trained specifically from its Xtrain
  for cohort, dataset in cohort_to_dataset.items():
    clf = cohort_to_clf.get(cohort)
    if clf is not None:
      train_pred, test_pred = clf.predict(dataset.Xtrain), clf.predict(dataset.Xtest)
      md_name = get_clf_name(clf)
      label_to_xgb_cls = None
      if 'XGB' in md_name:
        xgb_cls = sorted(set(dataset.ytrain))
        label_to_xgb_cls = {i: xgb_cls[i] for i in range(len(xgb_cls))}
        train_pred = np.array([label_to_xgb_cls[lb] for lb in train_pred]).astype(np.float)
        test_pred = np.array([label_to_xgb_cls[lb] for lb in test_pred]).astype(np.float)
      if not set(train_pred).issubset(np.unique(dataset.ytrain).astype(np.float)):
        print(label_to_xgb_cls)
        print(cohort, 'training', set(train_pred), set(dataset.ytrain))
      if not set(test_pred).issubset(np.unique(dataset.ytrain).astype(np.float)):
        print(label_to_xgb_cls)
        print('!!![c9]', cohort, 'test', set(test_pred), set(dataset.ytest))
      train_score_dict = MyScorer.apply_scorers(scorers, dataset.ytrain, train_pred)
      train_perf_df = append_perf_row_generic(
        train_perf_df, train_score_dict, {'Xtype': 'train', 'Cohort': cohort, 'Model': md_name,
                                          'Count': dataset.Xtrain.shape[0], 'Trial': trial_i, 'Year': year_label})
      test_score_dict = MyScorer.apply_scorers(scorers, dataset.ytest, test_pred)
      test_perf_df = append_perf_row_generic(
        test_perf_df, test_score_dict, {'Xtype': 'test', 'Cohort': cohort, 'Model': md_name,
                                        'Count': dataset.Xtest.shape[0], 'Trial': trial_i, 'Year': year_label})
  return train_perf_df, test_perf_df


def show_best_clf_per_cohort(perf_df: pd.DataFrame, Xtype, sort_by='accuracy'):
  # overall_acc by trial --> mean, std
  print(f'{Xtype} set performance by cohort:')

  # Obtain overall perf eval (groupby trial and model and compile cohort perf into 1 row for overall perf)
  no_cohort_col_list = perf_df.columns.to_list()
  no_cohort_col_list.remove('Cohort')
  no_cohort_col_list.remove('Xtype')
  overall_perf_df = pd.DataFrame(columns=no_cohort_col_list)
  groupby_trial_model = perf_df.groupby(by=['Trial', 'Model'])
  for tr_md, cohort_perf in groupby_trial_model:
    Xsize = cohort_perf['Count'].sum()
    overall_acc = np.dot(cohort_perf[SCR_ACC].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_acc_err1 = np.dot(cohort_perf[SCR_ACC_ERR1].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_underpred = np.dot(cohort_perf[SCR_UNDERPRED].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_overpred = np.dot(cohort_perf[SCR_OVERPRED].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_rmse = np.sqrt(np.dot(cohort_perf[SCR_RMSE].to_numpy()**2, cohort_perf['Count'].to_numpy()) / Xsize)
    overall_perf_df = overall_perf_df.append({'Trial': tr_md[0], 'Model': tr_md[1], 'Count': Xsize,
                                              SCR_ACC: overall_acc, SCR_ACC_ERR1: overall_acc_err1,
                                              SCR_UNDERPRED: overall_underpred, SCR_OVERPRED: overall_overpred,
                                              SCR_RMSE: overall_rmse}, ignore_index=True)
  overall_perf = pd.merge(overall_perf_df.groupby(by=['Model', 'Year']).mean().reset_index(),
                          overall_perf_df.groupby(by=['Model', 'Year']).std().reset_index(),
                          on=['Model', 'Year'],
                          how='left',
                          suffixes=('_mean', '_std')).dropna(axis=1)

  # Obtain best classifier for each cohort by their average accuracy
  clf_perf_mean_std = pd.merge(perf_df.groupby(by=['Cohort', 'Model', 'Year']).mean().reset_index(),
                               perf_df.groupby(by=['Cohort', 'Model', 'Year']).std().reset_index(),
                               on=['Cohort', 'Model', 'Year'],
                               how='left',
                               suffixes=('_mean', '_std'))
  best_clf_perf_idx = clf_perf_mean_std.groupby('Cohort')['accuracy_mean'].transform(max) == clf_perf_mean_std['accuracy_mean']
  best_clf_perf = clf_perf_mean_std[best_clf_perf_idx].drop_duplicates(subset=['Cohort', 'accuracy_mean'], keep='first')
  best_clf_xsize = best_clf_perf['Count_mean'].sum()
  print(f'Mean overall {Xtype} size: ', best_clf_xsize)
  print('Mean overall accuracy of best clf for each cohort: ',
        np.dot(best_clf_perf['accuracy_mean'], best_clf_perf['Count_mean']) / best_clf_xsize)

  # Obtain overall performance of the best classifiers for each cohort put together
  best_overall_df = perf_df.join(best_clf_perf.set_index(['Cohort', 'Model', 'Year']), on=['Cohort', 'Model', 'Year'], how='inner')
  no_cohort_col_list.remove('Model')
  best_overall_perf = pd.DataFrame(columns=no_cohort_col_list)
  for trial, cohort_perf in best_overall_df.groupby('Trial'):
    Xsize = cohort_perf['Count'].sum()
    overall_acc = np.dot(cohort_perf[SCR_ACC].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_acc_err1 = np.dot(cohort_perf[SCR_ACC_ERR1].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_underpred = np.dot(cohort_perf[SCR_UNDERPRED].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_overpred = np.dot(cohort_perf[SCR_OVERPRED].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_rmse = np.sqrt(np.dot(cohort_perf[SCR_RMSE].to_numpy() ** 2, cohort_perf['Count'].to_numpy()) / Xsize)
    best_overall_perf = best_overall_perf.append({'Trial': trial, 'Count': Xsize,
                                                  SCR_ACC: overall_acc, SCR_ACC_ERR1: overall_acc_err1,
                                                  SCR_UNDERPRED: overall_underpred, SCR_OVERPRED: overall_overpred,
                                                  SCR_RMSE: overall_rmse}, ignore_index=True)
  best_overall_perf = best_overall_perf.dropna(axis=1)

  best_clf_perf_styler = format_perf_df(best_clf_perf.sort_values(by=sort_by+'_mean', ascending=False))
  overall_perf_styler = format_perf_df(overall_perf)
  #best_overall_perf_styler = format_perf_df(best_overall_perf)
  print(pd.DataFrame({'Mean': best_overall_perf.mean(), 'Std': best_overall_perf.std()}))
  return best_clf_perf, overall_perf, best_clf_perf_styler, overall_perf_styler, best_overall_perf

