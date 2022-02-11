import pandas as pd
from pathlib import Path


def gen_surg_ckey_mapping_df(main_data_df: pd.DataFrame):
  surg_ckeys_mapping_df = pd.DataFrame(main_data_df['SURG_CASE_KEY'])
  surg_ckeys_mapping_df['PSEUDO_SURG_CASE_KEY'] = pd.Series(surg_ckeys_mapping_df['SURG_CASE_KEY'].index).add(30000)
  return surg_ckeys_mapping_df


def replace_with_new_surg_ckey(data_df: pd.DataFrame, surg_ckeys_mapping_df: pd.DataFrame, dfname=''):
  print("\nOriginal %s df shape: " % dfname, data_df.shape)
  new_df = data_df.join(surg_ckeys_mapping_df.set_index('SURG_CASE_KEY'), on='SURG_CASE_KEY', how='inner')  # TODO: inner or right?
  new_df['SURG_CASE_KEY'] = new_df['PSEUDO_SURG_CASE_KEY']
  new_df.drop(columns=['PSEUDO_SURG_CASE_KEY'], inplace=True)
  print("Updated-surg-ckey df shape: ", new_df.shape)

  return new_df


def remove_mrn(data_df: pd.DataFrame, dfname=''):
  if 'MRN' in data_df.columns:
    new_data_df = data_df.drop(columns=['MRN'])
    print("\nRemoved MRN in %s df" % dfname)
    return new_data_df
  return data_df


# TODO: distinguish hist and out-of-sample from save_dir!!
def deidentify_model_input_data(dashb_fp, cpt_fp, ccsr_fp, med_fp, dtime_fp=None, save_dir=None):
  # Use new indexing as pseudo surg_case_key, replacing the original ones
  dashb_df = pd.read_csv(dashb_fp)
  surg_ckeys_mapping_df = gen_surg_ckey_mapping_df(dashb_df)

  # Load the auxiliary dataframes
  cpt_df = pd.read_csv(cpt_fp)
  ccsr_df = pd.read_csv(ccsr_fp)
  med_df = pd.read_csv(med_fp)
  dtime_df = pd.read_csv(dtime_fp) if dtime_fp is not None else None
  all_dfs = {'Dashboard': dashb_df, 'CPT': cpt_df, 'CCSR': ccsr_df, 'Medication': med_df, 'DateTime': dtime_df}

  # Remove MRN column
  new_dfs = {}
  for dfname, df in all_dfs.items():
    if df is not None:
      new_df = remove_mrn(df, dfname)
      new_df = replace_with_new_surg_ckey(new_df, surg_ckeys_mapping_df, dfname)
      new_dfs[dfname] = new_df

  if save_dir:
    data_dir = Path(save_dir)
    surg_ckeys_mapping_df\
      .rename(columns={'SURG_CASE_KEY': 'ORIGINAL_SURG_CASE_KEY',
                       'PSEUDO_SURG_CASE_KEY': 'SURG_CASE_KEY'}, inplace=True)\
      .to_csv(data_dir / 'surg_case_key_og2pseudo.csv', index=False)
    for dfname, df in new_dfs.items():
      df.to_csv(data_dir / f'{dfname}.csv', index=False)

  return new_dfs, surg_ckeys_mapping_df


if __name__ == '__main__':
  data_dir = Path('../Data_new_all/ModelInput')
  save_dir = Path('../Data_new_all/ModelInput-cloud')

  # Historical Data
  new_dfs, surg_ckeys_mapping_df = deidentify_model_input_data(
    data_dir / 'historic.csv', data_dir / 'cpt_hist.csv', data_dir / 'ccsr_hist.csv', data_dir / 'medication.csv'
  )
  print(surg_ckeys_mapping_df.head(10))
  surg_ckeys_mapping_df.to_csv('./tmp_map.csv')
  print(new_dfs)

  # Out-of-sample Test Data
  new_dfs, surg_ckeys_mapping_df = deidentify_model_input_data(
    data_dir / 'outsample.csv', data_dir / 'cpt_os.csv', data_dir / 'ccsr_os.csv', data_dir / 'rx_os.csv'
  )
  print(surg_ckeys_mapping_df.head(10))
  surg_ckeys_mapping_df.to_csv('./tmp_map.csv')
  print(new_dfs)

  pd.DataFrame({'id': [100, 103, 105], 'score': [4,3,5]}).to_csv("./test.csv")

