import pandas as pd
import numpy as np
from typing import Any, Dict, List
from IPython.display import display

from copy import deepcopy
from globals import *
from c1_data_preprocessing import Dataset
from c4_model_perf import MyScorer, format_perf_df, get_clf_name
from utils_eval import *


# TODO: update show_best_xxxx: 1. change function name, 2. exclude surgeon perf from model perf

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
                                     'Model': SURGEON, 'Trial': trial_i, 'Year': year_label
                                     })
  return perf_df


def eval_cohort_clf(cohort_to_dataset: Dict[str, Dataset], cohort_to_clf: Dict[str, Any], scorers: List = None,
                    trial_i=None, sda_only=False, surg_only=False, years=None, train_perf_df=None, test_perf_df=None):
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
    # TODO: enable surg_only, sda_only flags
    Xtrain, ytrain = dataset.get_Xytrain_by_case_key(dataset.train_case_keys,
                                                     sda_only=sda_only, surg_only=surg_only, years=years)
    Xtest, ytest = dataset.get_Xytest_by_case_key(dataset.test_case_keys,
                                                  sda_only=sda_only, surg_only=surg_only, years=years)
    clf = cohort_to_clf.get(cohort)
    if (clf is not None) and (Xtrain.shape[0] > 0 and Xtest.shape[0] > 0):
      train_pred, test_pred = clf.predict(Xtrain), clf.predict(Xtest)
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
        print('!!![c9]', cohort, 'test', set(test_pred), set(ytest))
      # apply and save eval scores
      train_score_dict = MyScorer.apply_scorers(scorers, ytrain, train_pred)
      train_perf_df = append_perf_row_generic(
        train_perf_df, train_score_dict, {**get_class_count(ytrain),
                                          **{'Xtype': 'train', 'Cohort': cohort, 'Model': md_name,
                                             'Count': Xtrain.shape[0], 'Trial': trial_i, 'Year': year_label}})
      test_score_dict = MyScorer.apply_scorers(scorers, ytest, test_pred)
      test_perf_df = append_perf_row_generic(
        test_perf_df, test_score_dict, {**get_class_count(ytest),
                                        **{'Xtype': 'test', 'Cohort': cohort, 'Model': md_name,
                                           'Count': Xtest.shape[0], 'Trial': trial_i, 'Year': year_label}})
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
    perf_df = perf_df.loc[perf_df['Model'].isin(models)]  # TODO: separate out surgeon pred from aggregation

  # Obtain overall perf eval (groupby trial and model and compile cohort perf into 1 row for overall perf)
  no_cohort_col_list = perf_df.columns.to_list()
  no_cohort_col_list.remove('Cohort')
  no_cohort_col_list.remove('Xtype')
  overall_perf_df = pd.DataFrame(columns=no_cohort_col_list)
  groupby_trial_model_year = perf_df.groupby(by=['Trial', 'Model', 'Year'])
  for tr_md_yr, cohort_perf in groupby_trial_model_year:
    overall_score_dict = agg_cohort_perf_scores(cohort_perf)
    overall_perf_df = overall_perf_df.append({**{'Trial': tr_md_yr[0], 'Model': tr_md_yr[1], 'Year': tr_md_yr[2]},
                                              **overall_score_dict}, ignore_index=True)
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
    overall_score_dict = agg_cohort_perf_scores(cohort_perf)
    best_overall_perf = best_overall_perf.append({**{'Trial': trial_yr[0], 'Year': trial_yr[1]},
                                                  **overall_score_dict}, ignore_index=True)
  best_overall_perf = best_overall_perf.dropna(axis=1)

  print('\n**Best clf for each cohort: ')
  display(format_perf_df(best_clf_perf.sort_values(by=sort_by+'_mean', ascending=False)))
  print('\n**Overall performance: ')
  display(format_perf_df(overall_perf))
  print('\n**Best clfs Overall performance:')
  best_overall_perf_df = pd.DataFrame({'Mean': best_overall_perf.mean(), 'Std': best_overall_perf.std()})
  # formatters = {'Count': lambda x: f'{x:.1f}', SCR_ACC: lambda x: f'{x:.1%}', SCR_ACC_ERR1: lambda x: f'{x:.1%}',
  #               SCR_OVERPRED2: lambda x: f'{x:.1%}', SCR_UNDERPRED2: lambda x: f'{x:.1%}', SCR_RMSE: lambda x: f'{x:.2f}'}
  display(format_row_wise(best_overall_perf_df.style, {**{'Count': "{:.1f}".format}, **SCR_FORMATTER}))
  return best_clf_perf, overall_perf, best_overall_perf


# Obtain overvall performance scores by aggregating over all cohorts' performance scores
def agg_cohort_perf_scores(cohort_perf: pd.DataFrame) -> Dict:
  Xsize, cohort_count = cohort_perf['Count'].sum(), cohort_perf['Count'].to_numpy()
  overall_acc = np.dot(cohort_perf[SCR_ACC].to_numpy(), cohort_perf['Count'].to_numpy()) / Xsize
  overall_acc_err1 = np.dot(cohort_perf[SCR_ACC_ERR1].to_numpy(), cohort_count) / Xsize
  overall_underpred2 = np.dot(cohort_perf[SCR_UNDERPRED2].to_numpy(), cohort_count) / Xsize
  overall_overpred2 = np.dot(cohort_perf[SCR_OVERPRED2].to_numpy(), cohort_count) / Xsize
  overall_underpred0 = np.dot(cohort_perf[SCR_UNDERPRED0].to_numpy(), cohort_count) / Xsize
  overall_overpred0 = np.dot(cohort_perf[SCR_OVERPRED0].to_numpy(), cohort_count) / Xsize
  overall_rmse = np.sqrt(np.dot(cohort_perf[SCR_RMSE].to_numpy() ** 2, cohort_count) / Xsize)
  return {'Count': Xsize, SCR_ACC: overall_acc, SCR_ACC_ERR1: overall_acc_err1,
          SCR_UNDERPRED0: overall_underpred0, SCR_OVERPRED0: overall_overpred0,
          SCR_UNDERPRED2: overall_underpred2, SCR_OVERPRED2: overall_overpred2,
          SCR_RMSE: overall_rmse}
