import argparse
import boto3
import numpy as np
import pandas as pd
import pathlib
import warnings

from collections import defaultdict
from imblearn.over_sampling import SMOTENC
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier

from . import globals, c6_surgeon, utils, c5_model_perf as md_perf
from . import c0_data_prepare as dp, c1_data_preprocessing as dpp
from .c1_data_preprocessing import Dataset
from .c3_ensemble import Ensemble
from .c4_model_eval import ModelPerf
from .c7_feature_engineering import FeatureEngineeringModifier, DecileGenerator
from .c8_models import *


### Goal: hyper parameter tuning for models with different feature sets
### Input: preprocessed df with feature fully engineered

def get_args():
  parser = argparse.ArgumentParser(description='Hyperparam Tuning Script')
  # platform
  parser.add_argument('--platform', '-pf', default='local', choices=['aws', 'local', 'gcp'], type=str)
  parser.add_argument('--data_dir', default=pathlib.Path('../ModelInput'), type=pathlib.Path)
  parser.add_argument('--device', '-d', default='cpu', choices=['cpu', 'cuda:0'], type=str)  # ineffective for now

  # dataset
  parser.add_argument('--cohort', default='all', type=str)
  parser.add_argument('--val_pct', default=0.2, type=float)
  parser.add_argument('--holdout_test', '--os_test', default=False, action='store_true')
  parser.add_argument('--skip_o2m', default=[True, True, True], nargs=3, type=bool)  # train, val, test
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
  parser.add_argument('--kfold', default=5, type=int)  # If 1, no cross validation
  parser.add_argument('--models', nargs='+')  # [all, lgr, svc, knn, dt, rmf, xgb]
  parser.add_argument('--ensemble', nargs='+')
  parser.add_argument('--md_dir', '--model_dir', type=pathlib.Path)  # directory to load / save models
  parser.add_argument('--scorers', default=[], nargs='+')  # [acc, acc_err1, overpred, underpred, bal_acc]
  # TODO: Have an arg for filepath of model params?   arg for where results are saved?

  args = parser.parse_args()
  return args


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


def load_decile_features(code_abbr, arg_ftrs, col2decile_ftrs2aggf):
  # NOTE: default aggregation function is 'max' for all decile-related features
  # TODO: Need to figure out a way to specify aggregation function on multiple decile features
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


def make_feature_cols(args):
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
    feature_cols.extend(globals.OS_CODES)

  return feature_cols, col2decile_ftrs2aggf, onehot_cols, discretize_cols


if __name__ == '__main__':
  # Get args
  args = get_args()

  # Load datasets
  dashb_fp, cpt_fp, ccsr_fp, med_fp, cpt_grp_fp, os_fp, os_cpt_fp, os_ccsr_fp, os_med_fp = get_all_data_fp(args)
  dashb_df = dp.prepare_data(dashb_fp, cpt_fp, cpt_grp_fp, ccsr_fp, med_fp)

  # Load feature column list
  feature_cols, col2decile_ftrs2aggf, onehot_cols, discretize_cols = make_feature_cols(args)

  # Generate a preprocessed Dataset with the designated features engineered
  if args.kfold == 1:
    datasets = [
      Dataset(dashb_df, args.outcome, feature_cols, col2decile_ftrs2aggf, onehot_cols,
              test_pct=args.test_pct, discretize_cols=discretize_cols, cohort=args.cohort, remove_o2m=args.skip_o2m[:2])
    ]
  elif args.kfold > 1:
    datasets = utils.gen_kfolds_datasets(dashb_df, args.kfold, feature_cols, shuffle_df=True, outcome=args.outcome,
                                         onehot_cols=onehot_cols, discretize_cols=discretize_cols,
                                         col2decile_ftr2aggf=col2decile_ftrs2aggf, cohort=args.cohort,
                                         remove_o2m=args.skip_o2m[:2])
  else:
    raise ValueError("kfold must be a positive int!")

  # Apply modeling
  for md in args.models:
    perf_df = md_perf.init_perf_df(md)
    for dataset in datasets:
      if args.val_pct == 0:
        train_model(None, {})
      elif args.val_pct < 1:
        tune_model(None, dataset.Xtrain, dataset.ytrain, None, args.kfold)
      else:
        continue
      # Model evaluation


  # Save actual features, performance tables & best models



  # Hold-out Test
  if args.os_test:
    if args.val_pct > 0:
      # Retrain the best model on the full dataset, and then apply on test set
      dataset = Dataset(dashb_df, args.outcome, feature_cols, col2decile_ftrs2aggf, onehot_cols,
                        test_pct=args.test_pct, discretize_cols=discretize_cols, cohort=args.cohort,
                        remove_o2m=args.skip_o2m[:2])
      # TODO
    else:
      pass

    os_df = dp.prepare_data(os_fp, os_cpt_fp, cpt_grp_fp, os_ccsr_fp, os_med_fp)
    os_dataset = None

    # Predict

    # Surgeon prediction
