# Deployment script for applying pre-trained models
import joblib
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

from c0_data_prepare import prepare_data
from c1_data_preprocessing import Dataset
from c1_feature_engineering import FeatureEngineeringModifier
from c2_models import SafeOneClassWrapper
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


if __name__ == '__main__':
  # ---------------------------------------IMPORTANT NOTES: -------------------------------------------------
  # 1. Use data_fp = DATA_DIR / "outsample3_pseudo_deploy.csv" to simulate real testing environment
  # 2. In the simulated testing mode, os_dataset.ytest is assigned a placeholder value of 1000,
  #    since actual LoS and admission/discharge time are not available
  # ---------------------------------------------------------------------------------------------------------

  # 1. Generate test set dataframe with all sources of information combined
  os_data_df = prepare_data(data_fp=DATA_DIR / "outsample3.csv",
                            cpt_fp=DATA_DIR / "cpt_os3.csv",
                            cpt_grp_fp=CPT_TO_CPTGROUP_FILE,
                            ccsr_fp=DATA_DIR / "ccsr_os3.csv",
                            medication_fp=DATA_DIR / "medication_os3.csv",
                            chews_fp=DATA_HOME / "chews_raw/chews_os.csv",
                            exclude2021=False,
                            force_weight=False)
  print(f'[test_main] Loaded os_data_df! Shape: {os_data_df.shape}')

  # 2. Load Meta Data generated based on historical set
  FtrEngMod = load_FeatureEngModifier(FTR_ENG_MOD_FILE)
  O2M_df = load_o2m_df(O2M_FILE)

  # 3. Preprocess & Engineer Features on test data, using meta data from historical set --> Dataset() object
  os_dataset = Dataset(os_data_df,
                       outcome=NNT,
                       ftr_cols=FtrEngMod.feature_cols,
                       test_pct=1,
                       target_features=FtrEngMod.feature_names,
                       remove_o2m=(False, O2M_df),
                       ftr_eng=FtrEngMod)
  print(f'[test_main] Finished data preprocessing and feature engineering! '
        f'os_dataset.Xtest shape: {os_dataset.Xtest.shape}, os_dataset.ytest shape: {os_dataset.ytest.shape}')

  # 4. Load pretrained model
  Clf = load_pretrained_model(PRETRAINED_CLFS_DIR / 'xgbclf.joblib')
  print(f'[test_main] Loaded pretrained model!')

  # 5. Apply pre-trained models and output predictions
  os_predicted_nnt = Clf.predict(os_dataset.Xtest)
  print('[test_main] Predicted outcome on os_dataset.Xtest:\n', os_predicted_nnt)

  # 6. Check prediction accuracy (meaningless if in simulated testing mode)
  print(f'[test_main] Out-of-sample Test Accuracy: '
        f'{"{:.1%}".format(accuracy_score(os_dataset.ytest, os_predicted_nnt))}')
