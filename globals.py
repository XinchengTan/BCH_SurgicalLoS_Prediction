diaglabels = ['Cardiovascular', 'Digestive', 'Endocrine',
                'Genetic', 'Hematologic', 'Immunologic', 'Infectious',
                'Mental', 'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic',
                'Nutrition', 'Optic', 'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin',
                'Uncategorized', 'Urogenital']

DASHDATA_COLS = ['SURG_CASE_KEY', 'LENGTH_OF_STAY', 'SPS_PREDICTED_LOS', 'BOOKING_DATE',
                 'SEX_CODE', 'AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE', 'PROC_DECILE', 'Endocrine',
                 'Genetic', 'Hematologic', 'Immunologic', 'Infectious', 'Mental', 'Metabolic',
                 'Musculoskeletal', 'Neoplasm', 'Neurologic', 'Nutrition', 'Optic', 'Oral',
                 'Otic', 'Renal', 'Respiratory', 'Skin', 'Uncategorized', 'Urogenital']

DATETIME_COLS = ['SURG_CASE_KEY', 'CARE_CLASS', 'ICU_BED_NEEDED',
                 'ADMIT_DATE', 'DISCHARGE_DATE', 'SURGEON_START_DT_TM', 'SURGERY_END_DT_TM']  # 'PRIMARY_PROC',

FEATURE_COLS = ['SEX_CODE', 'AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE', 'PROC_DECILE', 'Endocrine',
                'Genetic', 'Hematologic', 'Immunologic', 'Infectious', 'Mental', 'Metabolic',
                'Musculoskeletal', 'Neoplasm', 'Neurologic', 'Nutrition', 'Optic', 'Oral',
                'Otic', 'Renal', 'Respiratory', 'Skin', 'Uncategorized', 'Urogenital']

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

LOS = "los"
NNT = "num_nights"
NNT_CUTOFFS = [1,2,3,4,5,6,7,8]

MAX_NNT = 7

SURGEON = 'SPS Surgeon'

LR = 'lr'
RIDGECV = 'ridgecv'
#QTR = 'qtr'
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

clf2name = {LGR: "Logistic Regression",
            SVC: "Support Vector Classifier",
            DTCLF: "Decision Tree Classifier",
            RMFCLF: "Random Forest Classifier",
            GBCLF: "Gradient Boosting Classifier",
            XGBCLF: "XGBoost Classifier"}


GMEAN = "G-mean"
F1 = "F1 score"
FPRPCT15 = "< 0.15"


SCR_ACC = 'accuracy'
SCR_AUC = 'roc_auc'
SCR_1NNT_TOL = 'Accuracy (tol = 1 NNT)'
SCR_1NNT_TOL_ACC = 'Accuracy (Hit rate and tol=1NNT)'
SCR_MULTI_ALL = 'multiple scorers (all)'