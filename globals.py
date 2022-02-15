from pathlib import Path

# Default built-in column names
SPS_PRED = 'SPS_PREDICTED_LOS'
LOS = "LENGTH_OF_STAY"
AGE = 'AGE_AT_PROC_YRS'
WEIGHT_ZS = 'WEIGHT_ZSCORE'
GENDER = 'SEX_CODE'
LANGUAGE = 'LANGUAGE_DESC'
INTERPRETER = 'INTERPRETER_NEED'
STATE = 'STATE_CODE'
REGION = 'MAJOR_REGION'
MILES = 'MILES_TRAVELED'
PROBLEM_CNT = 'PROBLEM_COUNT'
PRIMARY_PROC = 'PRIMARY_PROC'

# Customized new column names during dataframe preprocessing
NNT = "NUM_OF_NIGHTS"
CCSRS, CCSR, CCSR_DECILE = 'CCSRS', 'CCSR', 'CCSR_DECILE'
CPTS, CPT, CPT_DECILE = 'CPTS', 'CPT', 'CPT_DECILE'
CPT_GROUPS, CPT_GROUP = 'CPT_GROUPS', 'CPT_GROUP'
MED1, MED1_DECILE = 'MED1', 'MED1_DECILE'
MED2, MED2_DECILE = 'MED2', 'MED2_DECILE'
MED3, MED3_DECILE = 'MED3', 'MED3_DECILE'
PPROC, PPROC_DECILE = 'PPROC', 'PPROC_DECILE'


# Column list for feature selection
OS_CODES = ['Cardiovascular', 'Digestive', 'Endocrine',
            'Genetic', 'Hematologic', 'Immunologic', 'Infectious',
            'Mental', 'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic',
            'Nutrition', 'Optic', 'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin',
            'Uncategorized', 'Urogenital']
diaglabels = OS_CODES

DASHDATA_COLS = ['SURG_CASE_KEY', 'LENGTH_OF_STAY', 'SPS_PREDICTED_LOS', 'BOOKING_DATE',
                 GENDER, AGE, WEIGHT_ZS] + OS_CODES

DATETIME_COLS = ['SURG_CASE_KEY', 'CARE_CLASS', 'ICU_BED_NEEDED', PRIMARY_PROC,
                 'ADMIT_DATE', 'DISCHARGE_DATE', 'SURGEON_START_DT_TM', 'SURGERY_END_DT_TM',
                 'HAR_ADMIT_DATE', 'HAR_DISCHARGE_DATE']

DRUG_COLS = ['LEVEL1_DRUG_CLASS_NAME', 'LEVEL2_DRUG_CLASS_NAME', 'LEVEL3_DRUG_CLASS_NAME']
NON_NUMERIC_COLS = ['SURG_CASE_KEY', CPTS, CPT_GROUPS, PRIMARY_PROC, CCSRS, 'ICD10S'] + DRUG_COLS
CONTINUOUS_COLS = [AGE, WEIGHT_ZS]

# Feature columns
FEATURE_COLS_NO_WEIGHT = ['SURG_CASE_KEY', GENDER, AGE, CPTS, CPT_GROUPS, PRIMARY_PROC, CCSRS] \
                         + OS_CODES  # , 'ICD10S'

FEATURE_COLS = FEATURE_COLS_NO_WEIGHT + [WEIGHT_ZS] # , 'ICD10S'

#FEATURE_COLS_PPROC_NO_WEIGHT_OLD = FEATURE_COLS_NO_WEIGHT + [PPROC_DECILE] # , 'ICD10S'

FEATURE_COLS_NO_WEIGHT_STATE_LANG_INTERP = FEATURE_COLS_NO_WEIGHT + [STATE, LANGUAGE, INTERPRETER] # , 'ICD10S'
FEATURE_COLS_NO_WEIGHT_ALLMEDS = FEATURE_COLS_NO_WEIGHT + DRUG_COLS + \
                                 [STATE, LANGUAGE, INTERPRETER, REGION, PROBLEM_CNT, MILES]  # , 'ICD10S'

FEATURE_COLS_NO_DECILE = ['SURG_CASE_KEY', GENDER, AGE, WEIGHT_ZS, CPTS, CPT_GROUPS, PRIMARY_PROC, CCSRS] + OS_CODES  # , 'ICD10S'

FEATURE_COLS_NO_OS = ['SURG_CASE_KEY', GENDER, AGE, WEIGHT_ZS, CPTS, CPT_GROUPS, PRIMARY_PROC, CCSRS]  # , 'ICD10S'

FEATURE_COLS_SPS = FEATURE_COLS + [SPS_PRED]
FEATURE_COLS_NO_OS_SPS = FEATURE_COLS_NO_OS + [SPS_PRED]

# Default medical code to decile feature to aggregation function
DEFAULT_COL2DECILE_FTR2AGGF = {
  CPT: {CPT_DECILE: 'max', f'{CPT}_COUNT': 'max'},
  PPROC: {PPROC_DECILE: 'max', f'{PPROC}_COUNT': 'max'},
  MED3: {MED3_DECILE: 'max', f'{MED3}_COUNT': 'max'},
}
# MED1: {MED1_DECILE: 'max'},
# MED2: {MED2_DECILE: 'max'},

ONEHOT_COL2DTYPE = {
  PRIMARY_PROC: str,
  CPTS: list,
  CPT_GROUPS: list,
  CCSRS: list,
  DRUG_COLS[0]: list,
  DRUG_COLS[1]: list,
  DRUG_COLS[2]: list,
}

# All possible numeric features:
ALL_POSSIBLE_NUMERIC_COLS = [
  AGE, WEIGHT_ZS, MILES, PROBLEM_CNT,
  'PPROC_COUNT', 'CPT_COUNT', 'CPT_GROUP_COUNT', 'MED1_COUNT', 'MED2_COUNT', 'MED3_COUNT', 'CCSR_COUNT',
  'PPROC_SD', 'CPT_SD', 'CPT_GROUP_SD', 'MED1_SD', 'MED2_SD', 'MED3_SD', 'CCSR_SD',
  'PPROC_QT25', 'CPT_QT25', 'CPT_GROUP_QT25', 'MED1_QT25', 'MED2_QT25', 'MED3_QT25', 'CCSR_QT25',
  'PPROC_QT75', 'CPT_QT75', 'CPT_GROUP_QT75', 'MED1_QT75', 'MED2_QT75', 'MED3_QT75', 'CCSR_QT75',
  'PPROC_MIN', 'CPT_MIN', 'CPT_GROUP_MIN', 'MED1_MIN', 'MED2_MIN', 'MED3_MIN', 'CCSR_MIN',
  'PPROC_MAX', 'CPT_MAX', 'CPT_GROUP_MAX', 'MED1_MAX', 'MED2_MAX', 'MED3_MAX', 'CCSR_MAX',
]


# Cohort labels
COHORT_ALL = 'All Cases'
COHORT_TONSIL = 'Tonsillectomy'
COHORT_SPINE = 'Spinal Fusion'
COHORT_HIP = 'Hip'
COHORT_ORTHO = 'Orthopedics'
COHORT_NEUROLOGIC = 'Spinal Fusion - Neurologic'
COHORT_NON_NEUROLOGIC = 'Spinal Fusion - non-Neurologic'

# TODO: Update this, since primary proc has been updated!
COHORT_TO_PPROCS = {COHORT_TONSIL: {'TONSILLECTOMY WITH ADENOIDECTOMY, ORL', 'TONSILLOTOMY WITH ADENOIDECTOMY, ORL',
                                    'TONSILLECTOMY, ORL', 'ADENOIDECTOMY, ORL', 'TONSILLOTOMY, ORL',
                                    'TONSILLECTOMY, LINGUAL, ORL'},
                    COHORT_SPINE: {'SPINE, FUSION, POSTERIOR, THORACIC, ORTH', 'SPINE, FUSION, POSTERIOR, THORACIC TO LU',
                                   'SPINE, FUSION, POSTERIOR, THORACIC TO PE', 'SPINE, FUSION, POSTERIOR, LUMBAR TO SACR',
                                   'SPINE FUSION POSTERIOR, LUMBAR TO SACRAL', 'SPINE, FUSION W/PLIF, ORTHO',
                                   'SPINE, FUSION, POSTERIOR, CERVICAL TO TH', 'SPINE, FUSION, POSTERIOR, CERVICAL, ORTH',
                                   'zzSPINE FUSION POSTERIOR, LUMBAR, ORTHO'}}

CCSRS_TONSIL = {'Chronic respiratory insufficiency',
                'Epilepsy',
                 'Malacia of trachea or larynx',
                 'Down syndrome',
                 'Austism spectrum disorder',
                 'Cerebral palsy',
                 'Esophageal reflux',
                 'Enterostomy',
                 'Neurodevelopmental disorder',
                 'Chronic rhinitis',
                 'Asthma',
                 'Obesity',
                 'Hearing loss'}

CCSRS_SPINE = {'Chronic respiratory insufficiency',
              'Bladder dysfunction',
              'Dysphagia',
              'Anxiety disorder',
              'Esophageal reflux',
              'Enterostomy',
              'Intellectual disability',
              'Epilepsy',
              'Asthma',
              'Tracheostomy'}

COHORT_TO_CCSRS = {COHORT_TONSIL: CCSRS_TONSIL,
                   COHORT_SPINE: CCSRS_SPINE}

# according to CDC definition: https://www.cdc.gov/ncbddd/childdevelopment/positiveparenting/infants.html
AGE_BINS = [0, 0.25, 0.5, 1, 2, 3, 6, 9, 12, 15, 18, float('inf')]

DELTA = 1e-8
SEED = 998

ADMIT = "ADMIT"
DISCHARGE = "DISCHARGE"
IPSTART = "IPSTART"
STARTOS = "SOS"
ENDOS = "EOS"
WHEELIN = "WHEELIN"
WHEELOUT = "WHEELOUT"

HOUR = "H"
DAY = "D"
NIGHT = "N"

MAX_NNT = 5
NNT_CUTOFFS = list(range(MAX_NNT+1))
NNT_CLASS_CNT = len(NNT_CUTOFFS) + 1
NNT_CLASSES = list(range(MAX_NNT+2))
NNT_CLASS_LABELS = [str(i) for i in range(MAX_NNT + 1)] + ["%d+" % MAX_NNT]

# Dataset type
XTRAIN = 'train'
XTEST = 'test'
XVAL = 'val'
XAGREE = 'model & surgeon agree'
XDISAGREE = 'model & surgeon disagree' # can define anything reasonable
XDISAGREE1 = 'model & surgeon diagree by 1 night'
XDISAGREE2 = 'model & surgeon disagree by 2 nights'
XDISAGREE_GT2 = 'model & surgeon disagree by 2+ nights'
XALL_ONE2ONE = 'one-to-one (all)'  # TODO: denoise training data to include only one-to-one cases
XMAJ_ONE2ONE = 'one-to-one w/ majority filter'


# Denoise options
DENOISE_ONLY_TRAIN = 'denoise-only-train'
DENOISE_TRAIN_TEST0 = 'denoise-train-test0'  # denoise train, and denoise test from noise in train
DENOISE_TRAIN_TEST1 = 'denoise-train-test1'  # DENOISE_TRAIN_TEST0 and then further denoise test


DENOISE_ALL = 'all'
DENOISE_O2M = 'o2m'
DENOISE_PURE_DUP = 'pure-dups'
DENOISE_DEL_O2M = 'remove o2m'


SURGEON = 'SPS Surgeon'

# Model types
LR = 'lr-reg'
RIDGECV = 'ridgecv-reg'
DT = 'dt-reg'
RMF = 'rmf-reg'
GB = 'gb-reg'

reg2name = {LR: "Linear Regression",
            RIDGECV: "RidgeCV",
            DT: "Decision Tree",
            RMF: "Random Forest",
            GB: "Gradient Boosting",
            }

LGR = 'lgr'
PR = 'pr'
SVC = 'svc'
KNN = 'knn'
DTCLF = 'dt'
RMFCLF = 'rmf'
GBCLF = 'gb'
XGBCLF = 'xgb'
ORDCLF_LOGIT = 'ord-clf-logit'
ORDCLF_PROBIT = 'ord-clf-probit'
BAL_BAGCLF = 'bal-bagging'
ENSEMBLE_MAJ_EQ = 'ensemble-maj-eq'

#             SVC: "Support Vector Classifier",
clf2name = {LGR: "Logistic Regression",
            PR: "Poisson Regression",
            KNN: "K Nearest Neighbor",
            DTCLF: "Decision Tree Classifier",
            RMFCLF: "Random Forest Classifier",
            GBCLF: "Gradient Boosting Classifier",
            XGBCLF: 'XGBoost',
            BAL_BAGCLF: "Balanced Bagging Classifier"
            }
clf2name_eval = dict(clf2name)
clf2name_eval[ENSEMBLE_MAJ_EQ] = "Ensemble Model (maj - eq)"
clf2name_eval[SURGEON] = "Surgeon Prediction"

# XGBCLF: "XGBoost Classifier",
# ORDCLF_LOGIT: "Ordinal Classifier - Logit",
# ORDCLF_PROBIT: "Ordinal Classifier - Probit",


binclf2name = {LGR: "Logistic Regression",
               SVC: "Support Vector Classifier",
               DTCLF: "Decision Tree Classifier",
               RMFCLF: "Random Forest Classifier",
               GBCLF: "Gradient Boosting Classifier",
               BAL_BAGCLF: "Balanced Bagging Classifier"
               }  # XGBCLF: "XGBoost Classifier"

ALL_MODELS = set(clf2name.keys()).union(binclf2name.keys())

# Tasks
TASK_REG = 'reg'
TASK_MULTI_CLF = 'multiclf'
TASK_BIN_CLF = 'binclf'
# todo: update the following 3 vars to the 3 above
REG = 'reg'
MULTI_CLF = 'multiclf'
BIN_CLF = 'binclf'
TASK2Name = {TASK_REG: "Regression", TASK_MULTI_CLF: "Multi-class classification", TASK_BIN_CLF: "Binary classification"}
ALL_TASKS = set(TASK2Name.keys())

# Ensemble weighting mechanism


GMEAN = "G-mean"
F1 = "F1 score"
FPRPCT15 = "< 0.15"

# Evaluation metrics
SCR_ACC = 'accuracy'
SCR_AUC = 'roc_auc'
SCR_ACC_BAL = 'balanced_accuracy'
SCR_ACC_ERR1 = 'Accuracy (tol = 1 NNT)'
SCR_ACC_ERR2 = 'Accuracy (tol = 2 NNT)'
SCR_UNDERPRED = 'underprediction rate'
SCR_OVERPRED = 'overprediction rate'
# TODO: Remove these
SCR_1NNT_TOL_ACC = 'Accuracy (Hit rate and tol=1NNT)'
SCR_MULTI_ALL = 'multiple scorers (all)'


# File path of evaluation results
RESULTS = Path('./results')
RES_SPS_DATA_ALL = RESULTS / 'sps_data_all'
RES_SPS_DATA_AGREE = RESULTS / 'sps_data_agree'
RES_SPS_DATA_DISAGREE = RESULTS / 'sps_data_disagree'
RES_SPS_DATA_TONSIL = RESULTS / 'sps_data_tonsil'
RES_SPS_DATA_SPINE = RESULTS / 'sps_data_spine'
RES_SPS_DATA_HIP = RESULTS / 'sps_data_hip'
