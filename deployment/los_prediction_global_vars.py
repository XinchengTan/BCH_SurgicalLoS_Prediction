"""
This file defines all the global variables used by the LoS Prediction pipeline
"""
from pathlib import Path

# Dependency files' default file path
DEPLOY_DATA_DIR = Path('../Deployment_test_data')
DEPLOY_DEP_FILES_DIR = Path('../Deployment_dep_files')
PRETRAINED_MODELS_DIR = DEPLOY_DEP_FILES_DIR / 'pretrained_models'
SKIP_CASES_FP = DEPLOY_DEP_FILES_DIR / 'skip_cases.csv'
CPT_GROUP_FP = DEPLOY_DEP_FILES_DIR / 'cpt2group.csv'

# Macros of column names
LOS = 'LENGTH_OF_STAY'
NNT = 'NUM_OF_NIGHTS'
SURG_CASE_KEY = 'SURG_CASE_KEY'
SPS_LOS_FTR = 'SPS_PREDICTED_LOS'
CCSRS = 'CCSRS'
ICD10S = 'ICD10S'
CPTS = 'CPTS'

OS_CODES = ['Cardiovascular', 'Digestive', 'Endocrine',
            'Genetic', 'Hematologic', 'Immunologic', 'Infectious',
            'Mental', 'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic',
            'Nutrition', 'Optic', 'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin',
            'Uncategorized', 'Urogenital']

# Macros of column list for different usages
DASHDATA_COLS_DEPLOY_EVAL = ['SURG_CASE_KEY', 'LENGTH_OF_STAY', 'SPS_PREDICTED_LOS', 'PRIMARY_PROC',
                             'SEX_CODE', 'AGE_AT_PROC_YRS', 'PROC_DECILE',
                             'HAR_ADMIT_DATE', 'HAR_DISCHARGE_DATE', 'CSN'] + OS_CODES  # 'WEIGHT_ZSCORE',

DASHDATA_COLS_PREDICT = ['SURG_CASE_KEY', 'SPS_PREDICTED_LOS', 'PRIMARY_PROC',
                         'SEX_CODE', 'AGE_AT_PROC_YRS', 'PROC_DECILE', 'CSN'] + OS_CODES  # 'WEIGHT_ZSCORE',

FEATURE_COLS_NO_WEIGHT = ['SURG_CASE_KEY', 'SEX_CODE', 'AGE_AT_PROC_YRS', 'PROC_DECILE', CPTS, 'CPT_GROUPS',
                          'PRIMARY_PROC', CCSRS] + OS_CODES

FEATURE_COLS_PPROC_NO_WEIGHT = FEATURE_COLS_NO_WEIGHT + ['PPROC_DECILE']


NON_NUMERIC_COLS = ['SURG_CASE_KEY', CPTS, 'CPT_GROUPS', 'PRIMARY_PROC', CCSRS]  # non-numeric columns in feature_cols

AGE_BINS = [0, 0.25, 0.5, 1, 2, 3, 6, 9, 12, 15, 18, float('inf')]
WEIGHT_Z_BINS = [float('-inf'), -4, -2, -1, 1, 2, 4, float('inf')]

# Max number of night to consider as a prediction class
MAX_NNT = 5
NNT_CUTOFFS = list(range(MAX_NNT+1))
NNT_CLASS_CNT = len(NNT_CUTOFFS) + 1

# Model abbrv.
LGR = 'lgr'
SVC = 'svc'
KNN = 'knn'
DTCLF = 'dt-clf'
RMFCLF = 'rmf-clf'
GBCLF = 'gb-clf'
ENSEMBLE_MAJ_EQ = 'ensemble-maj-eq'

clf2name = {LGR: "Logistic Regression",
            KNN: "K Nearest Neighbor",
            DTCLF: "Decision Tree Classifier",
            RMFCLF: "Random Forest Classifier",
            GBCLF: "Gradient Boosting Classifier",
            }
ALL_CLFS = set(clf2name.keys())
ALL_MODELS = ALL_CLFS.union({ENSEMBLE_MAJ_EQ})

# Types of Prediction Task
TASK_REG = 'reg'
TASK_MULTI_CLF = 'multiclf'
TASK_BIN_CLF = 'binclf'
TASK2Name = {TASK_REG: "Regression", TASK_MULTI_CLF: "Multi-class classification", TASK_BIN_CLF: "Binary classification"}
ALL_TASKS = set(TASK2Name.keys())