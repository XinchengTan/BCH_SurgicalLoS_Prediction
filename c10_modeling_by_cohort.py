import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Hashable, Any

from globals import *
from c1_data_preprocessing import Dataset
from c4_model_perf import MyScorer, append_perf_row_generic, format_perf_df
from c2_models import get_model_by_cohort, SafeOneClassWrapper


# helper function to filter dashb_df by cohort count
def filter_df_by_cohort_count(dashb_df: pd.DataFrame,  cohort_col: str, min_cohort_size: int):
  cohort_groupby = dashb_df.groupby(cohort_col)
  if min_cohort_size > 1:
    filtered_cohort = cohort_groupby.filter(lambda x: len(x) >= min_cohort_size)
    return filtered_cohort
  return dashb_df


# Generate 'ktrials' of cohort-to-test-indices mapping, save the shuffled dashb_df as well
def gen_ktrial_cohort_test_idxs(dashb_df: pd.DataFrame, cohort_col: str, min_cohort_size=50, ktrials=10, test_pct=0.2):
  assert cohort_col in {SURG_GROUP, PRIMARY_PROC}, f'cohort_col must be one of [{SURG_GROUP, PRIMARY_PROC}]!'
  assert cohort_col in dashb_df.columns, f'{cohort_col} must exist in dashb_df columns!'

  if min_cohort_size is not None:
    assert min_cohort_size >= 0, 'min_cohort_size must be a non-negative int!'
    dashb_df = filter_df_by_cohort_count(dashb_df, cohort_col, min_cohort_size)

  kt_cohort2testIdxs = []  # [ dict{cohort: [idx0, idx1, ...]}, ...]
  kt_pprocDf = []
  for k in range(ktrials):
    # save shuffled df
    pproc_df = dashb_df.sample(frac=1).reset_index(drop=True)
    kt_pprocDf.append(pproc_df)
    # save test indices
    cohort2testIdxs = {}
    for pproc, pp_df in pproc_df.groupby(cohort_col):
      n = pp_df.shape[0]
      cohort2testIdxs[pproc] = np.random.choice(np.arange(n), int(n * test_pct), replace=False)
    kt_cohort2testIdxs.append(cohort2testIdxs)

  return kt_pprocDf, kt_cohort2testIdxs


def gen_pproc_ktrial_datasets(ktrials, decileFtr_config, kt_dfs, kt_cohort2testIdxs, test_pct, min_cohort_size):
  kt_datasets = []
  for k in range(ktrials):
    # Make dataset from existing train-test split cohort idxs
    dataset_k = gen_cohort_datasets(
      kt_dfs[k], PRIMARY_PROC, min_cohort_size, kt_cohort2testIdxs[k],
      outcome=NNT, ftr_cols=FEATURE_COLS_NO_WEIGHT_ALLMEDS,
      col2decile_ftrs2aggf=decileFtr_config, onehot_cols=[CCSRS],
      test_pct=test_pct, discretize_cols=['AGE_AT_PROC_YRS'], scaler='robust'
    )
    kt_datasets.append(dataset_k)
  return kt_datasets


def gen_cohort_datasets(dashb_df: pd.DataFrame, cohort_col: str, min_cohort_size=10, cohort2testIdxs=None,
                        **kwargs) -> Dict[Hashable, Dataset]:
  """
  Generates a mapping from cohort name to a Dataset object, with train & test set splitted.
  Decile-related features are computed only on the cohort cases.

  :param dashb_df: a pd.Dataframe object containing all sources of information
  :param cohort_col: Either SURG_GROUP or PRIMARY_PROC
  :param min_cohort_size: minimum number of samples per cohort
  :param cohort2testIdxs: a dict mapping cohort name to a numpy array of test indices to use for that cohort
  :param kwargs: input args for Dataset() instantiation

  :return: A mapping from cohort name to a Dataset object
  """
  assert cohort_col in {SURG_GROUP, PRIMARY_PROC}, f'cohort_col must be one of [{SURG_GROUP, PRIMARY_PROC}]!'
  assert cohort_col in dashb_df.columns, f'{cohort_col} must exist in dashb_df columns!'

  # Generate cohort to cohort_data_df mapping
  cohort_groupby = filter_df_by_cohort_count(dashb_df, cohort_col, min_cohort_size).groupby(cohort_col)
  print('[c10 gen_cohort_dataset] Effective sample size: ', cohort_groupby.size().sum())

  # Generate cohort to Dataset() mapping
  cohort_to_dataset = {}
  cases_in_use = 0
  for cohort, cohort_data_df in cohort_groupby:
    print('\n\n***Cohort: ', cohort, '\n\n')
    if cohort2testIdxs is not None:
      dataset = Dataset(cohort_data_df, test_idxs=cohort2testIdxs[cohort], **kwargs)
    else:
      dataset = Dataset(cohort_data_df, **kwargs)  # 'cohort_col' should not exist in onehot_cols in **kwargs

    # Leave out cohorts that are emptied due to 1. one-to-many cases cleaning 2. nan feature row dropping 3. unseen code
    if dataset.Xtrain.shape[0] > 0 and dataset.Xtest.shape[0] > 0:
      cohort_to_dataset[cohort] = dataset
      cases_in_use += dataset.Xtrain.shape[0] + dataset.Xtest.shape[0]

  print('=====================================================================================')
  print(f'*Total input cohorts: {len(cohort_groupby)}')
  print(f'*Actual cohort datasets: {len(cohort_to_dataset)}')
  print(f'*{cases_in_use} cases are used out of {dashb_df.shape[0]} available cases')
  print('=====================================================================================')
  return cohort_to_dataset


def train_cohort_clf(md, class_weight, cohort_to_dataset, train_sda_only=False):
  # TODO: add training only on SDA cases
  cohort_to_clf = {}
  for cohort, dataset in cohort_to_dataset.items():
    clf = get_model_by_cohort(md, class_weight)
    if md == XGBCLF:  # generate a mapping with continuous label class
      xgb_cls = sorted(set(dataset.ytrain))
      xgb_cls_to_label = {xgb_cls[i]: i for i in range(len(xgb_cls))}
      ytrain = np.array([xgb_cls_to_label[y] for y in dataset.ytrain])
    else:
      ytrain = np.array(dataset.ytrain)
    clf.fit(dataset.Xtrain, ytrain)
    cohort_to_clf[cohort] = clf
  return cohort_to_clf


def eval_cohort_clf(cohort_to_dataset: Dict[str, Dataset], cohort_to_clf: Dict[str, Any], scorers: Dict[str, Any] = None,
                    trial_i=None, train_perf_df=None, test_perf_df=None):
  if scorers is None:
    scorers = MyScorer.get_scorer_dict(DEFAULT_SCORERS)
  if train_perf_df is None:
    train_perf_df = pd.DataFrame(columns=['Trial', 'Xtype', 'Cohort', 'Model'] + list(scorers.keys()))
  if test_perf_df is None:
    test_perf_df = pd.DataFrame(columns=['Trial', 'Xtype', 'Cohort', 'Model'] + list(scorers.keys()))
  for cohort, dataset in cohort_to_dataset.items():
    clf = cohort_to_clf.get(cohort)
    if clf is not None:
      train_pred, test_pred = clf.predict(dataset.Xtrain), clf.predict(dataset.Xtest)
      md_name = clf.__class__.__name__ if not isinstance(clf, SafeOneClassWrapper) else clf.model_type
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
        print('!!![c10]', cohort, 'test', set(test_pred), set(dataset.ytest))
      train_score_dict = MyScorer.apply_scorers(scorers.keys(), dataset.ytrain, train_pred)
      train_perf_df = append_perf_row_generic(
        train_perf_df, train_score_dict, {'Xtype': 'train', 'Cohort': cohort, 'Model': md_name,
                                          'Count': dataset.Xtrain.shape[0], 'Trial': trial_i})
      test_score_dict = MyScorer.apply_scorers(scorers.keys(), dataset.ytest, test_pred)
      test_perf_df = append_perf_row_generic(
        test_perf_df, test_score_dict, {'Xtype': 'test', 'Cohort': cohort, 'Model': md_name,
                                        'Count': dataset.Xtest.shape[0], 'Trial': trial_i})
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
  overall_perf = pd.merge(overall_perf_df.groupby('Model').mean().reset_index(),
                          overall_perf_df.groupby('Model').std().reset_index(),
                          on=['Model'],
                          how='left',
                          suffixes=('_mean', '_std')).dropna(axis=1)

  # Obtain best classifier for each cohort by their average accuracy
  clf_perf_mean_std = pd.merge(perf_df.groupby(by=['Cohort', 'Model']).mean().reset_index(),
                               perf_df.groupby(by=['Cohort', 'Model']).std().reset_index(),
                               on=['Cohort', 'Model'],
                               how='left',
                               suffixes=('_mean', '_std'))
  best_clf_perf_idx = clf_perf_mean_std.groupby('Cohort')['accuracy_mean'].transform(max) == clf_perf_mean_std['accuracy_mean']
  best_clf_perf = clf_perf_mean_std[best_clf_perf_idx].drop_duplicates(subset=['Cohort', 'accuracy_mean'], keep='first')
  best_clf_xsize = best_clf_perf['Count_mean'].sum()
  print(f'Mean overall {Xtype} size: ', best_clf_xsize)
  print('Mean overall accuracy of best clf for each cohort: ',
        np.dot(best_clf_perf['accuracy_mean'], best_clf_perf['Count_mean']) / best_clf_xsize)

  # Obtain overall performance of the best classifiers for each cohort put together
  best_overall_df = perf_df.join(best_clf_perf.set_index(['Cohort', 'Model']), on=['Cohort', 'Model'], how='inner')
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


def cohort_modeling_ktrials(decileFtr_config, models, k_datasets, train_perf_df=None, test_perf_df=None):
  # print decile agg funcs
  for k, v in decileFtr_config.items():
    print(k, v)

  models = [LGR, KNN, RMFCLF, XGBCLF] if models is None else models  # GBCLF,
  for k in tqdm(range(len(k_datasets))):
    # Fit and eval models
    for md in models:
      print('md=', md)
      cohort_to_clf_pproc = train_cohort_clf(md=md, class_weight=None, cohort_to_dataset=k_datasets[k])
      train_perf_df, test_perf_df = eval_cohort_clf(
        k_datasets[k], cohort_to_clf_pproc, None, k, train_perf_df, test_perf_df
      )

  return train_perf_df, test_perf_df
