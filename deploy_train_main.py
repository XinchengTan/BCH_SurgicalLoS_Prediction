# Deployment script for training model on historical dataset
import joblib
import pandas as pd
from copy import deepcopy
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from typing import Dict, Iterable

from c0_data_prepare import prepare_data
from c1_data_preprocessing import Dataset
from c2_models import get_model
from globals import *
from globals_fs import *


# Returns a dict for decile feature aggregation function mapping
def get_decileFtr_config(verbose=False) -> Dict:
  # Modify the following to use different aggregation functions for different decile features
  decileFtr_config = deepcopy(DEFAULT_COL2DECILE_FTR2AGGF)
  decileFtr_config[PPROC]['PPROC_COUNT'] = 'sum'
  decileFtr_config[CPT]['CPT_COUNT'] = 'sum'
  decileFtr_config[CCSR]['CCSR_COUNT'] = 'sum'
  decileFtr_config[MED123]['MED123_COUNT'] = 'sum'
  decileFtr_config[CPT]['CPT_DECILE'] = 'sum'

  print('\n[train_main] Loaded decile feature aggregation config!\n')
  if verbose:
    for k, v in decileFtr_config.items():
      print(k, v)
  return decileFtr_config


# Initialize a dict of {model abbreviation: None} to save trained classifier object
def init_md_to_clf(md_list: Iterable) -> Dict:
  md_to_clf = {md: None for md in md_list}
  return md_to_clf


if __name__ == '__main__':
  # 1. Generate training set dataframe with all sources of information combined
  hist_data_df = prepare_data(data_fp=DATA_DIR / "historic3.csv",
                         cpt_fp=DATA_DIR / "cpt_hist.csv",
                         cpt_grp_fp=CPT_TO_CPTGROUP_FILE,
                         ccsr_fp=DATA_DIR / "ccsr_hist.csv",
                         medication_fp=DATA_DIR / "medication_hist.csv",
                         exclude2021=False,
                         force_weight=False)
  print(f'\n[train_main] Loaded os_data_df! Shape: {hist_data_df.shape}')

  # 2. Preprocess & Engineer Features on training data -> Dataset() object
  # To change the hypterparameters, e.g. what features to use, how to scale data, which feature to one-hot encode etc.,
  # update the args to Dataset() below
  hist_dataset = Dataset(df=hist_data_df, outcome=NNT,
                         ftr_cols=FEATURE_COLS_NO_WEIGHT_ALLMEDS,
                         col2decile_ftrs2aggf=get_decileFtr_config(),
                         onehot_cols=[CCSRS],
                         discretize_cols=['AGE_AT_PROC_YRS'],
                         scaler='robust', scale_numeric_only=True,
                         remove_o2m=(True, True),
                         test_pct=0)
  print(f'\n[train_main] Finished data preprocessing and feature engineering! '
        f'hist_dataset.Xtest shape: {hist_dataset.Xtrain.shape}, '
        f'hist_dataset.ytest shape: {hist_dataset.ytrain.shape}\n')

  # 4. Train models -- Modify the 'md_list' to add or remove models
  # Supported models can be found in c2_models.get_model()
  md_to_clf = init_md_to_clf(md_list=[LGR, KNN, RMFCLF, XGBCLF])
  for md in tqdm(md_to_clf):
    # 4.1 Train classifier
    print(f'[train_main] Start to train {md}')
    clf = get_model(md, cls_weight=None)
    clf.fit(hist_dataset.Xtrain, hist_dataset.ytrain)
    md_to_clf[md] = clf

    # 4.2 Save trained classifier
    with open(PRETRAINED_CLFS_DIR / f'{md}clf.joblib', 'wb') as md_file:
      joblib.dump(clf, md_file)
    print(f'[train_main] Saved "{md}" to "{md}clf.joblib"')

    # 4.3 Evaluate training performance
    hist_predicted_nnt = clf.predict(hist_dataset.Xtrain)
    print(f'[train_main] Training Accuracy of {md}: '
          f'{"{:.1%}".format(accuracy_score(hist_dataset.ytrain, hist_predicted_nnt))}')

  # 5. Save the pipeline meta data for separate testing
  hist_dataset.FeatureEngMod.save_to_pickle(FTR_ENG_MOD_FILE)
  print(f'\n[train_main] Saved FeatureEngineeringModifier to {FTR_ENG_MOD_FILE}!')

  pd.DataFrame(hist_dataset.o2m_df_train).to_csv(O2M_FILE, index=False)
  print(f'\n[train_main] Saved O2M Cases to {O2M_FILE}!')
