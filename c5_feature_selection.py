import numpy as np
import pandas as pd

import globals
from c1_data_preprocessing import Dataset
from jyp2_modeling import run_classifier


def feature_selection_ktrials(dataset: Dataset, ktrials=10):
  # change seed in every trial

  return


def feature_selection_single_trial(dashb_df: pd.DataFrame, rand_state, base_features=None, features_to_add=None):
  if base_features is None:
    base_features = [globals.PPROC_DECILE]
  if features_to_add is None:
    features_to_add = [
      globals.CPT_DECILE, globals.MED3_DECILE,
      f'{globals.PPROC}_COUNT', f'{globals.CPT}_COUNT', f'{globals.MED3}_COUNT',
      f'{globals.PPROC}_QT25', f'{globals.CPT}_QT25', f'{globals.MED3}_QT25',
      f'{globals.PPROC}_QT75', f'{globals.CPT}_QT75', f'{globals.MED3}_QT75',
      # f'{globals.PPROC}_SD', f'{globals.CPT}_SD', f'{globals.MED3}_SD',  (TODO: fix sd = 0 -- ddof=0, or fillna(0) )
      globals.AGE, globals.GENDER, globals.REGION, globals.STATE, globals.MILES,
      globals.LANGUAGE, globals.INTERPRETER
    ]
    features_to_add.extend(globals.OS_CODES)
    features_to_add.extend([globals.PRIMARY_PROC, globals.CPT, globals.CCSRS, globals.DRUG_COLS[2]])

  col2decile_ftrs2aggf = {
    globals.CPT: {
      globals.CPT_DECILE: 'max', f'{globals.CPT}_COUNT': 'max',
      f'{globals.CPT}_QT25': 'max', f'{globals.CPT}_QT75': 'max'
    },
    globals.PPROC: {
      globals.PPROC_DECILE: 'max', f'{globals.PPROC}_COUNT': 'max',
      f'{globals.PPROC}_QT25': 'max', f'{globals.PPROC}_QT75': 'max'
    },
    globals.MED3: {
      globals.MED3_DECILE: 'max', f'{globals.MED3}_COUNT': 'max',
      f'{globals.MED3}_QT25': 'max', f'{globals.MED3}_QT75': 'max'
    },
  }
  dataset_noScaler = Dataset(dashb_df, globals.NNT, ftr_cols=globals.FEATURE_COLS_NO_WEIGHT_ALLMEDS,
                             col2decile_ftrs2aggf=col2decile_ftrs2aggf, onehot_cols=[], discretize_cols=[globals.AGE],
                             test_pct=0.2, remove_o2m=[False, False], scaler=None)



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
