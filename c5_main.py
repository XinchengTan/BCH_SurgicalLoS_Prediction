import argparse
import boto3
import joblib
import numpy as np
import pandas as pd
import pathlib
import warnings

from collections import defaultdict

import globals, utils
import c0_data_prepare as dp
from c1_data_preprocessing import Dataset
from c3_ensemble import Ensemble
from c4_model_perf import *
from c1_feature_engineering import FeatureEngineeringModifier, DecileGenerator
from c2_models import *


### Goal: hyper parameter tuning for models with different feature sets
### Input: preprocessed df with feature fully engineered

def get_args():
  parser = argparse.ArgumentParser(description='Hyperparam Tuning Script')
  # platform
  parser.add_argument('--platform', '-pf', default='local', choices=['aws', 'local', 'gcp'], type=str)
  parser.add_argument('--data_dir', default=pathlib.Path('../ModelInput'), type=pathlib.Path)
  parser.add_argument('--param_dir', default=pathlib.Path('..'), type=pathlib.Path)  # dir to load model2params.json
  parser.add_argument('--device', '-d', default='cpu', choices=['cpu', 'cuda:0'], type=str)  # ineffective for now

  # dataset
  parser.add_argument('--cohort', default='all', type=str)
  parser.add_argument('--holdout_test', '--os_test', default=False, action='store_true')
  parser.add_argument('--skip_o2m', default=[True, True, True], nargs=3, type=bool)  # train, val, test
  parser.add_argument('--scaler', default='robust', type=str)
  parser.add_argument('--scale_nonnumeric', default=False, action='store_true')
  parser.add_argument('--outcome', default=globals.NNT, type=str)

  # features -- demographics
  parser.add_argument('--age', default='disc', choices=['disc', 'cont', 'none'], type=str)
  parser.add_argument('--gender', default=False, action='store_true')
  parser.add_argument('--weight', default='none', choices=['disc', 'cont', 'none'], type=str)
  parser.add_argument('--language', '--lang', default=False, action='store_true')
  parser.add_argument('--interpreter', '--interp', default=False, action='store_true')
  parser.add_argument('--state', default=False, action='store_true')
  parser.add_argument('--region', default=False, action='store_true')
  parser.add_argument('--miles', default=False, action='store_true')
  # features -- medical conditions
  parser.add_argument('--prob_cnt', default=False, action='store_true')
  ### The following args has default value = None; list item choice: none, oh, dcl, cnt, sd, min, max, qt25, qt75
  parser.add_argument('--pproc', nargs='+')
  parser.add_argument('--cpt', nargs='+')
  parser.add_argument('--cpt_grp', nargs='+')
  parser.add_argument('--ccsr', nargs='+')
  parser.add_argument('--med1', nargs='+')
  parser.add_argument('--med2', nargs='+')
  parser.add_argument('--med3', nargs='+')
  parser.add_argument('--os', nargs='+')  # only 'oh' is available
  # TODO: specify decile feature agg functions? workaround dict? Or sth like: dcl_avg, dcl_max ??

  # modeling
  parser.add_argument('--ktrial', default=5, type=int)  # If 1, just generate 1 test dataset
  parser.add_argument('--kfold', default=5, type=int)  # If 1, no cross validation
  parser.add_argument('--models', nargs='+')  # [all, lgr, svc, knn, dt, rmf, xgb]
  parser.add_argument('--ensemble', default=None, nargs='+')  # todo
  parser.add_argument('--md_dir', '--model_dir', type=pathlib.Path)  # directory to load models / save trained models
  parser.add_argument('--md_fnames', '--model_filenames', default=[], nargs='+')  # a list of model file names aligning with --models
  parser.add_argument('--result_dir', type=pathlib.Path)  # directory to save results (model perfs)
  parser.add_argument('--scorers', default=[], nargs='+')  # [acc, acc_err1, overpred, underpred, bal_acc]
  # TODO: Have an arg for filepath of model params?

  args = parser.parse_args()
  return args


# Get historic & out-of-sample data file path
def get_all_data_fp(args):
  if args.platform == 'aws':
    client = boto3.client('s3')
    # Historical dataset
    dashb_fp = client.get_object(Bucket='bch-data-cara', Key='historic_aws.csv')
    cpt_fp = client.get_object(Bucket='bch-data-cara', Key='cpt_hist_aws.csv')
    ccsr_fp = client.get_object(Bucket='bch-data-cara', Key='ccsr_hist_aws.csv')
    med_fp = client.get_object(Bucket='bch-data-cara', Key='medication_hist_aws.csv')
    cpt_grp_fp = client.get_object(Bucket='bch-data-cara', Key='cpt2group.csv')

    # Out-of-sample test dataset
    os_fp = client.get_object(Bucket='bch-data-cara', Key='outsample_aws.csv')
    os_cpt_fp = client.get_object(Bucket='bch-data-cara', Key='cpt_os_aws.csv')
    os_ccsr_fp = client.get_object(Bucket='bch-data-cara', Key='ccsr_os_aws.csv')
    os_med_fp = client.get_object(Bucket='bch-data-cara', Key='medication_os_aws.csv')

  elif args.platform == 'local':
    data_dir = args.data_dir
    # Historical dataset
    dashb_fp = data_dir / 'historic.csv'
    cpt_fp = data_dir / 'cpt_hist.csv'
    ccsr_fp = data_dir / 'ccsr_hist.csv'
    med_fp = data_dir / 'medication_hist.csv'
    cpt_grp_fp = data_dir / 'cpt2group.csv'

    # Out-of-sample test dataset
    os_fp = data_dir / 'outsample.csv'
    os_cpt_fp = data_dir / 'cpt_os.csv'
    os_ccsr_fp = data_dir / 'ccsr_os.csv'
    os_med_fp = data_dir / 'medication_os.csv'
  else:
    raise NotImplementedError
  return dashb_fp, cpt_fp, ccsr_fp, med_fp, cpt_grp_fp, os_fp, os_cpt_fp, os_ccsr_fp, os_med_fp


# Get Model abbr to a mapping of its hyperparameters (load from json)
def get_model2params(args):
  md_param_dir = args.param_dir
  assert md != globals.ENSEMBLE_MAJ_EQ
  param_fp = md_param_dir / 'model2params.json'
  with open(param_fp) as json_file:
    params = json.load(json_file)
  return params


# Get decile feature mapping
# TODO: Need to figure out a way to specify aggregation function on multiple decile features
def load_decile_features(code_abbr, arg_ftrs, col2decile_ftrs2aggf):
  # NOTE: default aggregation function is 'max' for all decile-related features
  if 'dcl' in arg_ftrs:
    col2decile_ftrs2aggf[code_abbr][f'{code_abbr}_DECILE'] = 'max'
  if 'cnt' in arg_ftrs:
    col2decile_ftrs2aggf[code_abbr][f'{code_abbr}_COUNT'] = 'max'
  if 'sd' in arg_ftrs:
    col2decile_ftrs2aggf[code_abbr][f'{code_abbr}_SD'] = 'max'
  if 'min' in arg_ftrs:
    col2decile_ftrs2aggf[code_abbr][f'{code_abbr}_MIN'] = 'max'
  if 'max' in arg_ftrs:
    col2decile_ftrs2aggf[code_abbr][f'{code_abbr}_MAX'] = 'max'
  if 'qt25' in arg_ftrs:
    col2decile_ftrs2aggf[code_abbr][f'{code_abbr}_QT25'] = 'max'
  if 'qt75' in arg_ftrs:
    col2decile_ftrs2aggf[code_abbr][f'{code_abbr}_QT75'] = 'max'

  return col2decile_ftrs2aggf


# Parse args to obtain feature-related args to preprocess & feature-engineer Dataset
def make_feature_cols(args):  # TODO: make folder name here!
  feature_cols = []
  discretize_cols = []
  onehot_cols = []
  col2decile_ftrs2aggf = defaultdict(dict)

  if args.age == 'disc':
    feature_cols.append(globals.AGE)
    discretize_cols.append(globals.AGE)
  elif args.age == 'cont':
    feature_cols.append(globals.AGE)

  if args.gender:
    feature_cols.append(globals.GENDER)

  if args.weight == 'disc':
    feature_cols.append(globals.WEIGHT_ZS)
    discretize_cols.append(globals.WEIGHT_ZS)
  elif args.weight == 'cont':
    feature_cols.append(globals.WEIGHT_ZS)

  # lodation
  if args.state:
    feature_cols.append(globals.STATE)
  if args.region:
    feature_cols.append(globals.REGION)
  if args.miles:
    feature_cols.append(globals.MILES)  # TODO: convert foreign to some random large number, or avg of other non-0 foreign miles??

  # language
  if args.lang:
    feature_cols.append(globals.LANGUAGE)
  if args.interpreter:
    feature_cols.append(globals.INTERPRETER)

  # medical conditions
  if args.prob_cnt:
    feature_cols.append(globals.PROBLEM_CNT)

  if args.pproc:
    feature_cols.append(globals.PRIMARY_PROC)
    if 'oh' in args.pproc:
      onehot_cols.append(globals.PRIMARY_PROC)
    col2decile_ftrs2aggf = load_decile_features(globals.PPROC, args.pproc, col2decile_ftrs2aggf)

  if args.cpt:
    feature_cols.append(globals.CPTS)
    if 'oh' in args.cpt:
      onehot_cols.append(globals.CPTS)
    col2decile_ftrs2aggf = load_decile_features(globals.CPT, args.cpt, col2decile_ftrs2aggf)

  if args.cpt_grp:
    feature_cols.append(globals.CPT_GROUPS)
    if 'oh' in args.cpt_grp:
      onehot_cols.append(globals.CPT_GROUPS)
    col2decile_ftrs2aggf = load_decile_features(globals.CPT_GROUP, args.cpt_grp, col2decile_ftrs2aggf)

  if args.ccsr:
    feature_cols.append(globals.CCSRS)
    if 'oh' in args.ccsr:
      onehot_cols.append(globals.CCSRS)
    col2decile_ftrs2aggf = load_decile_features(globals.CCSR, args.ccsr, col2decile_ftrs2aggf)

  if args.med1:
    feature_cols.append(globals.DRUG_COLS[0])
    if 'oh' in args.med1:
      onehot_cols.append(globals.DRUG_COLS[0])
    col2decile_ftrs2aggf = load_decile_features(globals.MED1, args.med1, col2decile_ftrs2aggf)

  if args.med2:
    feature_cols.append(globals.DRUG_COLS[1])
    if 'oh' in args.med2:
      onehot_cols.append(globals.DRUG_COLS[1])
    col2decile_ftrs2aggf = load_decile_features(globals.MED2, args.med2, col2decile_ftrs2aggf)

  if args.med3:
    feature_cols.append(globals.DRUG_COLS[2])
    if 'oh' in args.med3:
      onehot_cols.append(globals.DRUG_COLS[2])
    col2decile_ftrs2aggf = load_decile_features(globals.MED3, args.med3, col2decile_ftrs2aggf)

  if args.os:
    feature_cols.extend(globals.OS_CODE_LIST)

  return feature_cols, col2decile_ftrs2aggf, onehot_cols, discretize_cols


# Load pre-trained model, return a mapping from model abbr to model object
def load_pretrained_models(args):
  mds = set(args.models)
  if len(mds - globals.ALL_MODELS) > 0:
    raise NotImplementedError("Input model set contains a model that is not implemented yet!")
  if len(args.md_fnames) == 0:
    return {}
  assert len(mds) == len(args.md_fnames), 'Each model must have a corresponding file name!'

  md2model_obj = {}
  for md, md_filename in zip(args.models, args.md_fnames):
    md2model_obj[md] = [joblib.load(args.model_dir / (md_filename + '.joblib'))]
  # TODO: update ensemble args in the future
  if args.ensemble is not None and len(args.ensemble) > 0:
    md2task2clf = defaultdict(lambda: defaultdict(list))
    for md, md_filename in zip(args.models, args.md_fnames):
      md2task2clf[md][globals.TASK_MULTI_CLF] = [joblib.load(args.model_dir / (md_filename + '.joblib'))]
    md2model_obj['ensemble'] = Ensemble(tasks=[globals.TASK_MULTI_CLF], md2clfs=md2task2clf)
  return md2model_obj


if __name__ == '__main__':
  # Get args
  args = get_args()

  # Load datasets
  dashb_fp, cpt_fp, ccsr_fp, med_fp, cpt_grp_fp, os_fp, os_cpt_fp, os_ccsr_fp, os_med_fp = get_all_data_fp(args)
  dashb_df = dp.prepare_data(dashb_fp, cpt_fp, cpt_grp_fp, ccsr_fp, med_fp)

  # Load feature column list
  feature_cols, col2decile_ftrs2aggf, onehot_cols, discretize_cols = make_feature_cols(args)

  # Generate preprocessed Dataset(s) with the designated features engineered
  if args.ktrial == 1:
    datasets = [
      Dataset(dashb_df, args.outcome, feature_cols, col2decile_ftrs2aggf, onehot_cols,
              test_pct=args.test_pct, discretize_cols=discretize_cols, cohort=args.cohort, remove_o2m=args.skip_o2m[:2])
    ]
  elif args.ktrial > 1:
    datasets = utils.gen_kfolds_datasets(dashb_df, args.ktrial, feature_cols, shuffle_df=True, outcome=args.outcome,
                                         onehot_cols=onehot_cols, discretize_cols=discretize_cols,
                                         col2decile_ftr2aggf=col2decile_ftrs2aggf, cohort=args.cohort,
                                         remove_o2m=args.skip_o2m[:2])
  else:
    raise ValueError("ktrial must be a positive int!")

  # Apply modeling
  # scorers: Acc, Acc_err1, Acc_err2, overpred rate, underpred rate
  md2params = get_model2params(args)
  scorers = MyScorer.get_scorer_dict([globals.SCR_ACC, globals.SCR_ACC_BAL, globals.SCR_ACC_ERR1, globals.SCR_ACC_ERR2,
                                      globals.SCR_OVERPRED, globals.SCR_UNDERPRED, globals.SCR_RMSE])
  cv_results = defaultdict(list)  # md: [cv_result_df1, cv_result_df2, ...] --- len() = ktrial
  perf_df = pd.DataFrame(columns=['Trial', 'Model'] + list(scorers.keys()))
  surg_perf = pd.DataFrame(columns=['Trial', 'Model'] + list(scorers.keys()))  # Trial, train_scores ...
  for trial, dataset in enumerate(datasets):  # ktrials
    # Surgeon prediction
    train_scores_dict, test_scores_dict, confmat_test = eval_surgeon_perf(dataset, scorers, False)
    surg_perf = append_perf_row_surg(surg_perf, trial, train_scores_dict)

    # Train / Tune models
    for md in args.models:
      if args.kfold == 1:
        clf = train_model(md, md2params[md], dataset.Xtrain, dataset.ytrain)
      else:
        grid_search = train_model_cv(md, dataset.Xtrain, dataset.ytrain, args.kfold, scorers=scorers,
                                     refit=globals.SCR_ACC_BAL)
        cv_results[md] = grid_search.cv_results_
        clf = grid_search.best_estimator_

      # Evaluate trained model / best estimator on Xtest
      if dataset.Xtest is not None and dataset.Xtest.shape[0] > 0:
        scores_row_dict = MyScorer.apply_scorers(scorers.keys(), dataset.ytest, clf.predict(dataset.ytest))
        perf_df = append_perf_row(perf_df, trial, md, scores_row_dict)
        perf_df = append_perf_row(perf_df, trial, 'Surgeon-test', test_scores_dict)

  # Save actual features, performance tables & best models, surgeon performance, (model_params) -- locally
  # TODO: Make a custom dir name for result_dir, based on features used -- or uuid?
  result_dir = pathlib.Path(args.result_dir / '#Feature_abbr???#')
  result_dir.mkdir(parents=True, exist_ok=True)
  surg_perf.to_csv(result_dir / 'surgeon_perf_train.csv', index=False)
  perf_df.to_csv(result_dir / 'model&surg_perf_test.csv', index=False)
  for md, md_cv_results in cv_results.items():
    for trial, md_cv_result in enumerate(md_cv_results):
      pd.DataFrame(md_cv_result).to_csv(result_dir / f'{trial}-{md}_cv_result.csv', index=False)
  for trial, dataset in enumerate(datasets):
    pd.DataFrame(columns=dataset.feature_names).to_csv(result_dir / f'{trial}-feature_names.csv', index=False)

  # Hold-out Test
  if args.os_test:
    # Load pretrained model / load selected model params & retrain
    md2model_obj = load_pretrained_models(args)
    # Retrain the best model on the full dataset, and then apply on test set
    hist_dataset = Dataset(dashb_df, args.outcome, feature_cols, col2decile_ftrs2aggf, onehot_cols,
                           test_pct=0, discretize_cols=discretize_cols, cohort=args.cohort,
                           remove_o2m=args.skip_o2m[:2], scaler=args.scaler,
                           scale_numeric_only=not args.scale_nonnumeric)
    # Load model parameters and train on full dataset
    if len(md2model_obj) == 0:
      md_params = get_model2params(args)
      for md, md_params in md_params.items():
        md2model_obj[md] = train_model(md, md_params, hist_dataset.Xtrain, hist_dataset.ytrain)

    # Generate holdout test data
    os_df = dp.prepare_data(os_fp, os_cpt_fp, cpt_grp_fp, os_ccsr_fp, os_med_fp)
    os_dataset = Dataset(os_df, args.outcome, feature_cols, col2decile_ftrs2aggf, onehot_cols,
                        test_pct=1, discretize_cols=discretize_cols, cohort=args.cohort,
                        decile_gen=hist_dataset.FeatureEngineer.decile_generator,
                        target_features=hist_dataset.feature_names,
                        remove_o2m=(False, hist_dataset.o2m_df_train),
                        scaler=hist_dataset.input_scaler)

    # Predict on out-of-sample dataset


    # Surgeon prediction

