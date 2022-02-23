import argparse

import c0_data_prepare as dp
import utils
from feature_selection_mains import feature_selection as ftr_select
from c2_models import get_model
from globals import *
from globals_fs import *


if __name__ == '__main__':
  # Parse args
  parser = argparse.ArgumentParser(description='Stepwise Batch Feature Enhancement')
  parser.add_argument('--scaler', default=None, type=str)
  parser.add_argument('--discretize', '--discr', default=None, nargs='+')  # age, miles, weight
  parser.add_argument('--ktrials', '--kt', default=5, type=int)
  args = parser.parse_args()

  # Prepare dataset
  data_df = dp.prepare_data(DATA_DIR / "historic3.csv",
                            DATA_DIR / "cpt_hist.csv",
                            DATA_HOME / "cpt2group.csv",
                            DATA_DIR / "ccsr_hist.csv",
                            DATA_DIR / "medication_hist.csv",
                            exclude2021=False,
                            force_weight=False)
  print('Prepared dataset shape: ', data_df.shape)

  # Define Feature selection Experiment Details
  suffix = f'kt{args.ktrials}'
  disc_cols = []
  if args.discretize is None:
    disc_cols = None
    suffix += '_discrNA'
  else:
    suffix += '_discr'
    for col in disc_cols:
      col = col.lower()
      if col == 'age':
        disc_cols.append(AGE)
        suffix += 'Age'
      elif col == 'miles':
        disc_cols.append(MILES)
        suffix += 'Miles'
      elif col == 'weight':
        disc_cols.append(WEIGHT_ZS)
        suffix += 'Wt'
      else:
        continue
  suffix += '_' + str(args.scaler)
  print('Input Scaler: ', args.scaler)

  # Generate 'ktrials' of datasets
  #model_abbrs = [LGR, KNN, RMFCLF, XGBCLF]
  model_abbrs = [KNN]
  ktrial_datasets = utils.make_k_all_feature_datasets(data_df, k=args.ktrials,
                                                       onehot_cols=[PRIMARY_PROC, CPTS, CCSRS, DRUG_COLS[3]],
                                                       discretize_cols=disc_cols, scaler=args.scaler)
  # For each model, stepwise add features from training data, and evaluate performance
  for model in model_abbrs:
    print('Model %s' % model)
    clf = get_model(model)
    train_perf_df, test_perf_df = ftr_select.FeatureSelector.stepwise_batch_addition_fs(clf, ktrial_datasets)
    train_perf_df.to_csv(FS_RESULTS_TRAIN_DIR / f'{model}_{suffix}.csv', index=False)
    test_perf_df.to_csv(FS_RESULTS_TEST_DIR / f'{model}_{suffix}.csv', index=False)
