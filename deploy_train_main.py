# Deployment script for training model on historical dataset
import pandas as pd

from c0_data_prepare import prepare_data
from c1_data_preprocessing import Dataset
from c2_models import get_model
from globals import *
from globals_fs import *


if __name__ == '__main__':
  # Load historical dataframe

  # Preprocess & Engineer Features on historical data --> Dataset() object

  # Train models

  # Sanity check on training set performance

  pass