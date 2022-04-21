# Deployment script for applying pre-trained models
import joblib
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

from c0_data_prepare import prepare_data
from c1_data_preprocessing import Dataset
from c1_feature_engineering import FeatureEngineeringModifier
from c2_models import SafeOneClassWrapper
from deploy_train_main import MODEL_TO_SCALER
from globals import *
from globals_fs import *


# Load Meta Data
def load_FeatureEngModifier(fp) -> FeatureEngineeringModifier:
  with open(fp, 'rb') as file:
    FtrEngMod = pickle.load(file)
  return FtrEngMod


# Load skip cases (one-to-many)
def load_o2m_df(fp) -> pd.DataFrame:
  o2m_df = pd.read_csv(fp)
  return o2m_df


# Load pretrained model
def load_pretrained_model(fp) -> SafeOneClassWrapper:
  with open(fp, 'rb') as md_file:
    clf = joblib.load(md_file)
  return clf


def load_scaled_os_dataset(scaler):
  assert scaler in MODEL_TO_SCALER.values(), f'Scaler "{scaler}" must be applied on historical data first!'
  # Load Meta Data generated based on historical set
  FtrEngMod = load_FeatureEngModifier(DEPLOY_DEP_FILES_DIR / f'hist_FtrEngMod_SCAL{scaler}.pkl')
  O2M_df = load_o2m_df(DEPLOY_DEP_FILES_DIR / f'hist_o2m_cases_SCAL{scaler}.csv')

  # Build Dataset object based on meta data from historical dataset
  dataset = Dataset(os_data_df,
                    outcome=NNT,
                    ftr_cols=FtrEngMod.feature_cols,
                    test_pct=1,
                    target_features=FtrEngMod.feature_names,
                    remove_o2m=(False, O2M_df),
                    ftr_eng=FtrEngMod)
  print(f'[test_main] Finished data preprocessing and feature engineering! '
        f'os_dataset.Xtest shape: {dataset.Xtest.shape}, os_dataset.ytest shape: {dataset.ytest.shape}')
  print('[test_main] Class labels: ', np.unique(dataset.ytest))
  return dataset


if __name__ == '__main__':
  # ---------------------------------------IMPORTANT NOTES: -------------------------------------------------
  # 1. Use data_fp = DATA_DIR / "outsample3_pseudo_deploy.csv" to simulate real testing environment
  # 2. In the simulated testing mode, os_dataset.ytest is assigned a placeholder value of 1000,
  #    since actual LoS and admission/discharge time are not available
  # ---------------------------------------------------------------------------------------------------------

  # 1. Generate test set dataframe with all sources of information combined
  os_data_df = prepare_data(data_fp=DATA_DIR / "outsample4.csv",
                            cpt_fp=DATA_DIR / "cpt_os4.csv",
                            cpt_grp_fp=CPT_TO_CPTGROUP_FILE,
                            ccsr_fp=DATA_DIR / "ccsr_os4.csv",
                            medication_fp=DATA_DIR / "medication_os4.csv",
                            chews_fp=DATA_HOME / "chews_raw/chews_os.csv",
                            exclude2021=False,
                            force_weight=False)
  print(f'[test_main] Loaded os_data_df! Shape: {os_data_df.shape}')

  # 2. Load input scaler to a preprocessed Dataset object on holdout test set, using meta data from the
  # correspondingly-scaled historical set
  md = SVCLF  # XGBCLF
  os_dataset = load_scaled_os_dataset(MODEL_TO_SCALER[md])

  # 3. Load pretrained model
  Clf = load_pretrained_model(PRETRAINED_CLFS_DIR / f'{md}clf.joblib')
  print(f'[test_main] Loaded pretrained model!')

  # 4. Apply pre-trained models and output predictions
  os_predicted_nnt = Clf.predict(os_dataset.Xtest)
  print('[test_main] Predicted outcome on os_dataset.Xtest:\n', os_predicted_nnt)

  # 5. Check prediction accuracy (meaningless if in simulated testing mode)
  print(f'[test_main] Out-of-sample Test Accuracy: '
        f'{"{:.1%}".format(accuracy_score(os_dataset.ytest, os_predicted_nnt))}')
