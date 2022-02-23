import os

import c0_data_prepare as dp
import utils
from c3_ensemble import Ensemble
from c4_model_perf import MyScorer
import c5_feature_selection as ftr_select
from c7_feature_engineering import FeatureEngineeringModifier, DecileGenerator
from c2_models import get_model
from globals import *
from globals_fs import *


if __name__ == '__main__':
  # Prepare dataset
  data_df = dp.prepare_data(DATA_DIR / "historic3.csv",
                            DATA_DIR / "cpt_hist.csv",
                            DATA_HOME / "cpt2group.csv",
                            DATA_DIR / "ccsr_hist.csv",
                            DATA_DIR / "medication_hist.csv",
                            exclude2021=False,
                            force_weight=False)
  print('Prepared dataset shape: ', data_df.shape)

  # # Define Feature selection Experiment Details
  # ktrials = 10
  # n_ftr_val = [15]
  # model_abbrs = [XGBCLF]
  # # No scaling
  # print('No scaling')
  # ktrial_datasets_sfs = utils.make_k_all_feature_datasets(data_df, k=ktrials,
  #                                                      onehot_cols=[],
  #                                                      discretize_cols=None, scaler=None)
  # for model in model_abbrs:
  #   print('Model %s' % model)
  #   clf = get_model(XGBCLF)
  #   train_perf_df, test_perf_df, ftr2counts = ftr_select.feature_selection_with_eval(
  #     clf, ktrial_datasets_sfs, scorers=None, fs_scorer='accuracy', n_feature_values=n_ftr_val, fs_how='sfs')
  #   train_perf_df.to_csv(FS_RESULTS_DIR / f'sfs/{model}_fs{ktrials}_train_discNA_scalerRob.csv', index=False)
  #   test_perf_df.to_csv(FS_RESULTS_DIR / f'sfs/{model}_fs{ktrials}_test_discNA_scalerRob.csv', index=False)
  #   with open(FS_RESULTS_DIR / f'sfs/ftr2cnt_{model}.txt') as f:
  #     f.write(str(ftr2counts))
