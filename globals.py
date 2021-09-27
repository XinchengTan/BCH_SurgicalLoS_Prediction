diaglabels = ['Cardiovascular', 'Digestive', 'Endocrine',
                'Genetic', 'Hematologic', 'Immunologic', 'Infectious',
                'Mental', 'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic',
                'Nutrition', 'Optic', 'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin',
                'Uncategorized', 'Urogenital']

FEATURE_COLS = ['SEX_CODE', 'AGE_AT_PROC_YRS', 'WEIGHT_ZSCORE',
                'PROC_DECILE', 'Cardiovascular', 'Digestive',
                'Endocrine', 'Genetic', 'Hematologic', 'Immunologic', 'Infectious',
                'Mental', 'Metabolic', 'Musculoskeletal', 'Neoplasm', 'Neurologic',
                'Nutrition', 'Optic', 'Oral', 'Otic', 'Renal', 'Respiratory', 'Skin',
                'Uncategorized', 'Urogenital']

DELTA = 1e-8

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

MAX_NNT = 7

LR = 'lr'
RIDGECV = 'ridgecv'
DT = 'dt'
RMF = 'rmf'
GB = 'gb'

model2name = {LR: "Linear Regression",
              RIDGECV: "RidgeCV",
              DT: "Decision Tree",
              RMF: "Random Forest",
              GB: "Gradient Boosting"}