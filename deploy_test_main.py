# Deployment script for applying pre-trained models
import pandas as pd

from c0_data_prepare import prepare_data
from c1_data_preprocessing import Dataset
from c2_models import get_model
from globals import *
from globals_fs import *


if __name__ == '__main__':
  # Load test set dataframe

  # Preprocess & Engineer Features on test data, using meta data from historical set --> Dataset() object

  # Apply pre-trained models and output predictions

  pass