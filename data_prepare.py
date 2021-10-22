"""
This script contains helper functions to prepare the raw dataset pulled from DB.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from . import globals


def gen_cols_with_nan(df):
  res = df.isnull().sum().sort_values(ascending=False)
  return res[res.values[1] > 0]


def print_df_info(df, dfname="Dashboard"):
  print("%s columns:\n" % dfname, dashb_df.keys())
  print("Number of cases:", df.shape[0])
  print("Number of NaNs in each column:\n", gen_cols_with_nan(df))
  print("\n")


def prepare_data(data_fp, dtime_fp, exclude21=True):
  """
  Prepares the patient data over 2018 - 2020
  """
  # Load dashboard dataset
  dashb_df = pd.read_csv(data_fp, parse_dates=True)
  print_df_info(dashb_df, dfname='Dashboard')
  # Drop rows with NaN in weight z-score
  dashb_df = dashb_df[globals.DASHDATA_COLS].dropna(subset=['WEIGHT_ZSCORE'])
  print_df_info(dashb_df, dfname='Dashboard')


  # Load datetime dataset
  dtime_df = pd.read_csv(dtime_fp, parse_dates=True)
  print_df_info(dtime_df, dfname='Datetime')
  # Drop rows that are NaN in selected columns
  dtime_df = dtime_df[globals.DATETIME_COLS].dropna(subset=globals.DATETIME_COLS)
  print_df_info(dtime_df, dfname='Datetime')


  # compute the number of nights

  # Load CPT and combine with existing hierarchy


  # Exclude 2021 data by request
  if exclude21:
    pass

  return dashb_df


if __name__ == "__main__":
  dashb_df = prepare_data(Path("../Data_new_all", "dashboard_data_18to21.csv"),
                          Path("../Data_new_all", "dtime_data_18to21.csv"))
