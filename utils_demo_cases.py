import numpy as np
import pandas as pd

from c1_data_preprocessing import Dataset
from globals import *
from globals_fs import *


def gen_model_outputs(dataset: Dataset, model, cohortwise_perf_df=None, count=20, topK_ftrs=5):
  test_case_keys = np.random.choice(dataset.test_case_keys, count, replace=False)
  # Top k most important features
  return


def gen_data_inputs(dataset: Dataset, test_case_keys):
  # Condensed features
  data_input_df = pd.DataFrame(dataset.test_cohort_df).reset_index(drop=False).set_index('SURG_CASE_KEY')
  data_input_df = data_input_df.loc[test_case_keys,
                                    ['PATIENT_KEY', 'MRN', GENDER, AGE, SURG_GROUP, PRIMARY_PROC, ADMIT_DTM,
                                     MED1, MED2, MED3, MED123, CPTS, CPT_GROUPS, CCSRS]]
  # All features
  model_input_df = pd.DataFrame(dataset.get_Xytest_by_case_key(test_case_keys), columns=dataset.feature_names)

  # Combine the input_dfs above
  input_df = data_input_df.join(model_input_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='inner')

  return input_df


# # Jupyter notebook scratch
# rand_keys = np.random.choice(os_dataset.test_case_keys, 20, replace=False)
# rand_Xtest, rand_ytest = os_dataset.get_Xytest_by_case_key(rand_keys)
#
# rand_xgb_pred = xgb.predict(rand_Xtest)
# ftr_imps = md_perf.gen_feature_importance(xgb.base_estimator, XGBCLF, reg=False, ftrs=os_dataset.feature_names)
#
# rand_os_df = os_data_df.set_index('SURG_CASE_KEY').loc[rand_keys][['MRN', SURG_GROUP, SPS_PRED]]
# rand_os_df['Model Prediction'] = rand_xgb_pred
# rand_os_df.sort_values(by=SURG_GROUP).to_csv(RESULT_DIR / 'example_output.csv', index=False)
#
# rand_keys_ftrs = os_dataset.test_case_keys[np.in1d(os_dataset.test_case_keys, rand_keys)]
# rand_os_input_df = os_dataset.test_cohort_df.loc[
#     rand_keys, ['PATIENT_KEY', 'MRN', GENDER, AGE, ADMIT_DTM,
#                 MILES, STATE, REGION, LANGUAGE, INTERPRETER,
#                 PRIMARY_PROC, SURG_GROUP, CPTS, CPT_GROUPS,
#                 CCSRS] + DRUG_COLS
# ]
#
# rand_ftrs_df = pd.DataFrame(os_dataset.get_Xytest_by_case_key(rand_keys)[0], columns=os_dataset.feature_names)
# rand_ftrs_df = rand_ftrs_df[[tup[0] for tup in ftr_imps]].add_prefix('ftr_')
# rand_ftrs_df['SURG_CASE_KEY'] = rand_keys_ftrs
#
# rand_os_input_df = rand_os_input_df.join(rand_ftrs_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='inner')
# rand_os_input_df.sort_values(by=SURG_GROUP).to_csv(RESULT_DIR / 'example_inputs.csv', index=True)
#
