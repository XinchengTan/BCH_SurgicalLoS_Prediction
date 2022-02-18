import numpy as np
import pandas as pd
from typing import List, Dict, Optional

import globals
from c1_data_preprocessing import Dataset
from jyp2_modeling import run_classifier

# TODO 0. Double check for nans in miles traveld, state code
# TODO 1. Finish stepwise addition func & Generate result tables & performance
# TODO 2. Add CCSR deciles & Test
# TODO 3. Add violin plot across **pproc**, cpt, ccsrs
# TODO 4. Think about how to handle one-to-many cases in the stepwise addition process


class FeatureSelector(object):

  def __init__(self):
    pass

  @staticmethod
  def stepwise_addition():
    pass

  @staticmethod
  def simulated_annealing():
    pass

  @staticmethod
  def resampled_simulated_annealing():
    pass




def feature_selection_ktrials(dataset: Dataset, ktrials=10, scaler=None):
  # change seed in every trial OR generate k-fold datasets

  # TODO: update the following dict to
  col2decile_ftrs2aggf = {
    globals.CPT: {
      globals.CPT_DECILE: 'max', f'{globals.CPT}_COUNT': 'max',
      f'{globals.CPT}_MEDIAN': 'max',
      f'{globals.CPT}_QT25': 'max', f'{globals.CPT}_QT75': 'max'
    },
    globals.PPROC: {
      globals.PPROC_DECILE: 'max', f'{globals.PPROC}_COUNT': 'max',
      f'{globals.PPROC}_MEDIAN': 'max',
      f'{globals.PPROC}_QT25': 'max', f'{globals.PPROC}_QT75': 'max'
    },
    globals.MED3: {
      globals.MED3_DECILE: 'max', f'{globals.MED3}_COUNT': 'max',
      f'{globals.MED3}_MEDIAN': 'max',
      f'{globals.MED3}_QT25': 'max', f'{globals.MED3}_QT75': 'max'
    },
  }
  #

  return


def stepwise_addition_single_trial(clf,
                                   dataset: Dataset,
                                   base_features: Optional[List[str]],
                                   features_to_add: Optional[List[str]],
                                   rand_state=globals.SEED):
  """
  This function iteratively adds features, one at a time, to a pre-defined classifier trained on dataset.Xtrain
  and evaluated on dataset.Xtest.

  :param dataset:
  :param base_features:
  :param features_to_add:
  :param rand_state:
  :return:
  """
  if base_features is None:
    base_features = [globals.PPROC_DECILE]
  if features_to_add is None:
    features_to_add = [
      globals.CPT_DECILE, globals.MED3_DECILE,
      f'{globals.PPROC}_COUNT', f'{globals.CPT}_COUNT', f'{globals.MED3}_COUNT',
      f'{globals.PPROC}_MEDIAN', f'{globals.CPT}_MEDIAN', f'{globals.MED3}_MEDIAN',
      f'{globals.PPROC}_QT25', f'{globals.CPT}_QT25', f'{globals.MED3}_QT25',
      f'{globals.PPROC}_QT75', f'{globals.CPT}_QT75', f'{globals.MED3}_QT75',
      # f'{globals.PPROC}_SD', f'{globals.CPT}_SD', f'{globals.MED3}_SD',  (TODO: fix sd = 0 -- ddof=0, or fillna(0) )
      globals.AGE, globals.GENDER, globals.REGION, globals.STATE, globals.MILES,
      globals.LANGUAGE, globals.INTERPRETER
    ]
    features_to_add.extend(globals.OS_CODES)
    features_to_add.extend([globals.PRIMARY_PROC, globals.CPT, globals.CCSRS, globals.DRUG_COLS[2]])




  # 0. start with PProc decile as a base feature
  # 1. add other deciles one by one (cpt, med3, ccsr)
  # - 1.1 add code count
  # - 1.2 add qt25
  # - 1.3 add qt75
  # - 1.4 add code SD
  # 2. add age
  # 3. add gender
  # 4. add major region
  # 5. add state code
  # 6. add miles
  # 7. add language
  # 8. add interpreter needed
  # 9. add OS code
  # 10. add PProc indicators
  # 11. add CPT indicators
  # 12. add CCSR indicators
  # 13. add Med3 indicators

  # For each iteration, try 1. no input scaling  2. scale numeric features only  3. scale all -- RobustScaler

  # TODO: What to do with removing o2m along the step-wise addition???

  return
