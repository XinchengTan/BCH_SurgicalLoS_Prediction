from pathlib import Path

OS_CODES = ['Cardiovascular', 'Digestive', 'Endocrine',
            'Genetic', 'Hematologic', 'Immunologic', 'Infectious',
            'Mental', 'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic',
            'Nutrition', 'Optic', 'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin',
            'Uncategorized', 'Urogenital']
diaglabels = OS_CODES

DASHDATA_COLS = ['SURG_CASE_KEY', 'LENGTH_OF_STAY', 'SPS_PREDICTED_LOS', 'BOOKING_DATE',
                 'SEX_CODE', 'AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE', 'PROC_DECILE'] + OS_CODES

DATETIME_COLS = ['SURG_CASE_KEY', 'CARE_CLASS', 'ICU_BED_NEEDED', 'PRIMARY_PROC',
                 'ADMIT_DATE', 'DISCHARGE_DATE', 'SURGEON_START_DT_TM', 'SURGERY_END_DT_TM',
                 'HAR_ADMIT_DATE', 'HAR_DISCHARGE_DATE']

FEATURE_COLS_NO_WEIGHT = ['SURG_CASE_KEY', 'SEX_CODE', 'AGE_AT_PROC_YRS', 'PROC_DECILE',
                          'CPTS', 'CPT_GROUPS', 'PRIMARY_PROC', 'CCSRS'] + OS_CODES  # , 'ICD10S'

FEATURE_COLS = FEATURE_COLS_NO_WEIGHT + ['WEIGHT_ZSCORE'] # , 'ICD10S'

FEATURE_COLS_PPROC_NO_WEIGHT_OLD = FEATURE_COLS_NO_WEIGHT + ['PPROC_DECILE'] # , 'ICD10S'

FEATURE_COLS_PPROC_NO_WEIGHT = FEATURE_COLS_NO_WEIGHT + ['PPROC_DECILE', 'STATE_CODE', 'LANGUAGE_DESC', 'INTERPRETER_NEED'] # , 'ICD10S'

FEATURE_COLS_NO_DECILE = ['SURG_CASE_KEY', 'SEX_CODE', 'AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE',
                          'CPTS', 'CPT_GROUPS', 'PRIMARY_PROC', 'CCSRS'] + OS_CODES  # , 'ICD10S'

FEATURE_COLS_NO_OS = ['SURG_CASE_KEY', 'SEX_CODE', 'AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE',
                      'CPTS', 'CPT_GROUPS', 'PRIMARY_PROC', 'CCSRS']  # , 'ICD10S'

MED1 = 'MED1'
SPS_LOS_FTR = 'SPS_PREDICTED_LOS'
FEATURE_COLS_SPS = FEATURE_COLS + [SPS_LOS_FTR]
FEATURE_COLS_NO_OS_SPS = FEATURE_COLS_NO_OS + [SPS_LOS_FTR]
NON_NUMERIC_COLS = ['SURG_CASE_KEY', 'CPTS', 'CPT_GROUPS', 'PRIMARY_PROC', 'CCSRS', 'ICD10S']
CONTINUOUS_COLS = ['AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE']

COHORT_ALL = 'All Cases'
COHORT_TONSIL = 'Tonsillectomy'
COHORT_SPINE = 'Spinal Fusion'
COHORT_HIP = 'Hip'
COHORT_ORTHO = 'Orthopedics'
COHORT_NEUROLOGIC = 'Spinal Fusion - Neurologic'
COHORT_NON_NEUROLOGIC = 'Spinal Fusion - non-Neurologic'


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

LOS = "LENGTH_OF_STAY"
NNT = "NUM_OF_NIGHTS"
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
LR = 'lr'
RIDGECV = 'ridgecv'
DT = 'dt'
RMF = 'rmf'
GB = 'gb'

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
DTCLF = 'dt-clf'
RMFCLF = 'rmf-clf'
GBCLF = 'gb-clf'
XGBCLF = 'xgb-clf'
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
REG = 'reg'
MULTI_CLF = 'multiclf'
BIN_CLF = 'binclf'
TASK2Name = {REG: "Regression", MULTI_CLF: "Multi-class classification", BIN_CLF: "Binary classification"}
ALL_TASKS = set(TASK2Name.keys())

# Ensemble weighting mechanism


GMEAN = "G-mean"
F1 = "F1 score"
FPRPCT15 = "< 0.15"

# Evaluation metrics
SCR_ACC = 'Accuracy'
SCR_AUC = 'roc_auc'
SCR_1NNT_TOL = 'Accuracy (tol = 1 NNT)'
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
