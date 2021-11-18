diaglabels = ['Cardiovascular', 'Digestive', 'Endocrine',
                'Genetic', 'Hematologic', 'Immunologic', 'Infectious',
                'Mental', 'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic',
                'Nutrition', 'Optic', 'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin',
                'Uncategorized', 'Urogenital']

DASHDATA_COLS = ['SURG_CASE_KEY', 'LENGTH_OF_STAY', 'SPS_PREDICTED_LOS', 'BOOKING_DATE',
                 'SEX_CODE', 'AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE', 'PROC_DECILE',
                 'Endocrine', 'Genetic', 'Hematologic', 'Immunologic', 'Infectious', 'Mental',
                 'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic', 'Nutrition', 'Optic',
                 'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin', 'Uncategorized', 'Urogenital']

DATETIME_COLS = ['SURG_CASE_KEY', 'CARE_CLASS', 'ICU_BED_NEEDED', 'PRIMARY_PROC',
                 'ADMIT_DATE', 'DISCHARGE_DATE', 'SURGEON_START_DT_TM', 'SURGERY_END_DT_TM']

FEATURE_COLS = ['SURG_CASE_KEY', 'SEX_CODE', 'AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE', 'PROC_DECILE',
                'Endocrine', 'Genetic', 'Hematologic', 'Immunologic', 'Infectious', 'Mental',
                'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic', 'Nutrition', 'Optic',
                'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin', 'Uncategorized', 'Urogenital',
                'CPT_GROUPS', 'PRIMARY_PROC', 'CCSRS']

FEATURE_COLS_NO_DECILE = ['SURG_CASE_KEY', 'SEX_CODE', 'AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE',
                'Endocrine', 'Genetic', 'Hematologic', 'Immunologic', 'Infectious', 'Mental',
                'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic', 'Nutrition', 'Optic',
                'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin', 'Uncategorized', 'Urogenital',
                'CPT_GROUPS', 'PRIMARY_PROC', 'CCSRS']

SPS_LOS_FTR = 'SPS_PREDICTED_LOS'
FEATURE_COLS_SPS = FEATURE_COLS + [SPS_LOS_FTR]
NON_NUMERIC_COLS = ['SURG_CASE_KEY', 'CPT_GROUPS', 'PRIMARY_PROC', 'CCSRS']


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
MAX_NNT = 7
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
SVC = 'svc'
DTCLF = 'dt-clf'
RMFCLF = 'rmf-clf'
GBCLF = 'gb-clf'
XGBCLF = 'xgb-clf'
ORDCLF_LOGIT = 'ord-clf-logit'
ORDCLF_PROBIT = 'ord-clf-probit'

clf2name = {LGR: "Logistic Regression",
            SVC: "Support Vector Classifier",
            DTCLF: "Decision Tree Classifier",
            RMFCLF: "Random Forest Classifier",
            GBCLF: "Gradient Boosting Classifier",
            }
# XGBCLF: "XGBoost Classifier",
# ORDCLF_LOGIT: "Ordinal Classifier - Logit",
# ORDCLF_PROBIT: "Ordinal Classifier - Probit",


binclf2name = {LGR: "Logistic Regression",
            SVC: "Support Vector Classifier",
            DTCLF: "Decision Tree Classifier",
            RMFCLF: "Random Forest Classifier",
            GBCLF: "Gradient Boosting Classifier",
            } # XGBCLF: "XGBoost Classifier"

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