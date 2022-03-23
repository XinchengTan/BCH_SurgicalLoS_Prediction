# Defines file system related global variables
from pathlib import Path

PROJECT_DIR = Path('/Users/caratan/Desktop/ML4Health/Dashboard-Project')
DATA_HOME = PROJECT_DIR / 'Data_new_all'
DATA_TRIALS_HOME = PROJECT_DIR / 'Data_ktrials'
DATA_DIR = DATA_HOME / 'ModelInput'

DEPLOY_DATA_DIR = PROJECT_DIR / 'Deployment_test_data'
DEPLOY_DEP_FILES_DIR = PROJECT_DIR / 'Deployment_dep_files'
CPT_TO_CPTGROUP_FILE = DEPLOY_DEP_FILES_DIR / 'cpt2group.csv'
FTR_ENG_MOD_FILE = DEPLOY_DEP_FILES_DIR / 'hist_FtrEngMod.pkl'
O2M_FILE = DEPLOY_DEP_FILES_DIR / 'skip_cases.csv'

RESULT_DIR = PROJECT_DIR / 'Results'
COHORTWISE_RESULTS_DIR = RESULT_DIR / 'Cohortwise'
AGGREGATIVE_RESULTS_DIR = RESULT_DIR / 'Aggregative'
FS_RESULTS_DIR = PROJECT_DIR / 'FS_results'
FS_RESULTS_TRAIN_DIR = FS_RESULTS_DIR / 'stepwise/train'
FS_RESULTS_TEST_DIR = FS_RESULTS_DIR / 'stepwise/test'

