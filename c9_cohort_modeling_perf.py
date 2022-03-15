import pandas as pd
import numpy as np
from typing import Any, Dict, List
from IPython.display import display

from copy import deepcopy
from globals import *
from c1_data_preprocessing import Dataset
from c4_model_perf import MyScorer, append_perf_row_generic, format_perf_df, get_clf_name, get_default_perf_df, get_class_count, to_numeric_count_cols


# TODO: add class_count
# TODO: update show_best_xxxx: 1. change function name, 2. exclude surgeon perf from model perf


def get_year_label(years) -> str:
  if years is None:
    return f'All'
  elif len(years) > 1:
    return f'{min(years)} - {max(years)}'
  else:
    return str(years[0])


# If certain evaluation metrics does not apply, display -1
def get_placeholder_perf_scores_dict(scorers: List) -> Dict:
  return {s: -1.0 for s in scorers}


def add_surgeon_cohort_perf(cohort, clf, dataset: Dataset, Xtype, scorers, trial_i, years, perf_df):
  year_label = get_year_label(years)
  md_name = get_clf_name(clf)

  if Xtype == 'train':
    X, y, data_case_keys = dataset.Xtrain, dataset.ytrain, dataset.train_case_keys
  else:
    X, y, data_case_keys = dataset.Xtest, dataset.ytest, dataset.test_case_keys

  # Fetch surg_pred and true outcome
  surg_pred_true = dataset.get_surgeon_pred_df_by_case_key(data_case_keys, years=years)
  if surg_pred_true.empty:
    scores_dict = get_placeholder_perf_scores_dict(scorers)
  else:
    scores_dict = MyScorer.apply_scorers(scorers, surg_pred_true[dataset.outcome], surg_pred_true[SPS_PRED])
  # Append scores with info to perf_df
  perf_df = append_perf_row_generic(perf_df, scores_dict,
                                    {'Xtype': Xtype, 'Cohort': cohort, 'Count': surg_pred_true.shape[0],
                                     'Model': 'Surgeon', 'Trial': trial_i, 'Year': year_label
                                     })
  return perf_df


def eval_cohort_clf(cohort_to_dataset: Dict[str, Dataset], cohort_to_clf: Dict[str, Any], scorers: List = None,
                    trial_i=None, surg_only=False, years=None, train_perf_df=None, test_perf_df=None):
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
        train_perf_df, train_score_dict, {**get_class_count(dataset.ytrain),
                                          **{'Xtype': 'train', 'Cohort': cohort, 'Model': md_name,
                                             'Count': dataset.Xtrain.shape[0], 'Trial': trial_i, 'Year': year_label}})
      test_score_dict = MyScorer.apply_scorers(scorers, dataset.ytest, test_pred)
      test_perf_df = append_perf_row_generic(
        test_perf_df, test_score_dict, {**get_class_count(dataset.ytest),
                                        **{'Xtype': 'test', 'Cohort': cohort, 'Model': md_name,
                                           'Count': dataset.Xtest.shape[0], 'Trial': trial_i, 'Year': year_label}})
      if surg_only:
        train_perf_df = add_surgeon_cohort_perf(cohort, clf, dataset, 'train', scorers, trial_i, years, train_perf_df)
        test_perf_df = add_surgeon_cohort_perf(cohort, clf, dataset, 'test', scorers, trial_i, years, test_perf_df)

  train_perf_df = to_numeric_count_cols(train_perf_df)
  test_perf_df = to_numeric_count_cols(test_perf_df)
  return train_perf_df, test_perf_df


def summarize_cohortwise_modeling_perf(perf_df: pd.DataFrame, Xtype, models=None, sort_by='accuracy'):
  # overall_acc by trial --> mean, std
  print(f'{Xtype} set performance by cohort:')

  if models is not None:
    perf_df = perf_df.loc[perf_df['Model'].isin(models)]

  # Obtain overall perf eval (groupby trial and model and compile cohort perf into 1 row for overall perf)
  no_cohort_col_list = perf_df.columns.to_list()
  no_cohort_col_list.remove('Cohort')
  no_cohort_col_list.remove('Xtype')
  overall_perf_df = pd.DataFrame(columns=no_cohort_col_list)
  groupby_trial_model_year = perf_df.groupby(by=['Trial', 'Model', 'Year'])
  for tr_md_yr, cohort_perf in groupby_trial_model_year:
    Xsize = cohort_perf['Count'].sum()
    overall_acc = np.dot(cohort_perf[SCR_ACC].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_acc_err1 = np.dot(cohort_perf[SCR_ACC_ERR1].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_underpred = np.dot(cohort_perf[SCR_UNDERPRED].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_overpred = np.dot(cohort_perf[SCR_OVERPRED].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_rmse = np.sqrt(np.dot(cohort_perf[SCR_RMSE].to_numpy()**2, cohort_perf['Count'].to_numpy()) / Xsize)
    overall_perf_df = overall_perf_df.append({'Trial': tr_md_yr[0], 'Model': tr_md_yr[1], 'Year': tr_md_yr[2],
                                              'Count': Xsize, SCR_ACC: overall_acc, SCR_ACC_ERR1: overall_acc_err1,
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
  #display(best_clf_perf)
  best_clf_xsize = best_clf_perf['Count_mean'].sum()
  print(f'Mean overall {Xtype} size: ', best_clf_xsize)
  print('Mean overall accuracy of best clf for each cohort: ',
        np.dot(best_clf_perf['accuracy_mean'], best_clf_perf['Count_mean']) / best_clf_xsize)

  # Obtain overall performance of the best classifiers for each cohort put together
  best_overall_df = perf_df.join(best_clf_perf.set_index(['Cohort', 'Model', 'Year']),
                                 on=['Cohort', 'Model', 'Year'],
                                 how='inner')
  no_cohort_col_list.remove('Model')
  best_overall_perf = pd.DataFrame(columns=no_cohort_col_list)
  for trial_yr, cohort_perf in best_overall_df.groupby(by=['Trial', 'Year']):
    Xsize = cohort_perf['Count'].sum()
    overall_acc = np.dot(cohort_perf[SCR_ACC].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_acc_err1 = np.dot(cohort_perf[SCR_ACC_ERR1].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_underpred = np.dot(cohort_perf[SCR_UNDERPRED].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_overpred = np.dot(cohort_perf[SCR_OVERPRED].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
    overall_rmse = np.sqrt(np.dot(cohort_perf[SCR_RMSE].to_numpy() ** 2, cohort_perf['Count'].to_numpy()) / Xsize)
    best_overall_perf = best_overall_perf.append({'Trial': trial_yr[0], 'Year': trial_yr[1], 'Count': Xsize,
                                                  SCR_ACC: overall_acc, SCR_ACC_ERR1: overall_acc_err1,
                                                  SCR_UNDERPRED: overall_underpred, SCR_OVERPRED: overall_overpred,
                                                  SCR_RMSE: overall_rmse}, ignore_index=True)
  best_overall_perf = best_overall_perf.dropna(axis=1)

  print('\n**Best clf for each cohort: ')
  display(format_perf_df(best_clf_perf.sort_values(by=sort_by+'_mean', ascending=False)))
  print('\n**Overall performance: ')
  display(format_perf_df(overall_perf))
  print('\n**Best clfs Overall performance:')
  display(pd.DataFrame({'Mean': best_overall_perf.mean(), 'Std': best_overall_perf.std()}))
  return best_clf_perf, overall_perf, best_overall_perf

