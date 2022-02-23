# Defines file system related global variables
from pathlib import Path

PROJECT_DIR = Path('/Users/caratan/Desktop/ML4Health/Dashboard-Project')
DATA_HOME = PROJECT_DIR / 'Data_new_all'
DATA_DIR = DATA_HOME / 'ModelInput'
DEPLOY_DATA_DIR = PROJECT_DIR / 'Deployment_test_data'
DEPLOY_DEP_FILES_DIR = PROJECT_DIR / 'Deployment_dep_files'
FS_RESULTS_DIR = PROJECT_DIR / 'FS_results'
FS_RESULTS_TRAIN_DIR = FS_RESULTS_DIR / 'stepwise/train'
FS_RESULTS_TEST_DIR = FS_RESULTS_DIR / 'stepwise/test'

