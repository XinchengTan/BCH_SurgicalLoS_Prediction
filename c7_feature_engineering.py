import numpy as np
import pandas as pd
from pathlib import Path

from . import globals


class FeatureEngineeringModifier(object):

  def __init__(self, onehot_cols=None, onehot_dtypes=None, trimmed_ccsr=None, discretize_cols=None,
               col2decile_ftrs2aggf=globals.DEFAULT_COL2DECILE_FTR2AGGF, decile_outcome=globals.LOS):
    self.onehot_cols, self.onehot_dtypes = onehot_cols, onehot_dtypes
    self.trimmed_ccsr = trimmed_ccsr
    self.discretize_cols = discretize_cols
    self.col2decile_ftr2aggf = col2decile_ftrs2aggf
    self.decile_outcome = decile_outcome
    self.decile_generator = DecileGenerator()

  def add_temporal_feature_admit_hour(self, data_df: pd.DataFrame):
    # TODO: FINISH THIS
    return

  def discretize_columns_df(self, Xdf: pd.DataFrame, discretize_cols, inplace=False):
    if discretize_cols is None:
      return Xdf

    if not inplace:
      Xdf = Xdf.copy()
    # Modify data matrix with discretized columns by request
    for dis_col in discretize_cols:
      if dis_col not in Xdf.columns.to_list():
        raise Warning("%s is not in Xdf columns!" % dis_col)
      elif dis_col == 'AGE_AT_PROC_YRS':
        Xdf[dis_col] = pd.cut(Xdf[dis_col], bins=globals.AGE_BINS, labels=False, right=False, include_lowest=True)
      elif dis_col == 'WEIGHT_ZSCORE':
        weightz_bins = [float('-inf'), -4, -2, -1, 1, 2, 4, float('inf')]
        print("Weight z-score bins: ", weightz_bins)
        Xdf[dis_col] = pd.cut(Xdf[dis_col], bins=weightz_bins, labels=False, right=False, include_lowest=True)
      else:
        raise Warning("%s discretization is not available yet!" % dis_col)
    return Xdf

  def dummy_code_discrete_cols(self, Xdf: pd.DataFrame):
    # Gender
    Xdf.loc[(Xdf.SEX_CODE != 'F'), 'SEX_CODE'] = 0.0
    Xdf.loc[(Xdf.SEX_CODE == 'F'), 'SEX_CODE'] = 1.0

    Xdf_cols = Xdf.columns.to_list()
    # Interpreter need or not
    if 'INTERPRETER_NEED' in Xdf_cols:
      Xdf.loc[(Xdf.INTERPRETER_NEED == 'N'), 'INTERPRETER_NEED'] = 0.0
      Xdf.loc[(Xdf.INTERPRETER_NEED == 'Y'), 'INTERPRETER_NEED'] = 1.0

    # State code
    if 'STATE_CODE' in Xdf_cols:
      Xdf[['IN_STATE', 'OUT_OF_STATE_US', 'FOREIGN']] = 0.0
      Xdf.loc[(Xdf.STATE_CODE == 'MA'), 'IN_STATE'] = 1.0
      Xdf.loc[(Xdf.STATE_CODE == 'Foreign'), 'FOREIGN'] = 1.0
      Xdf.loc[((Xdf.IN_STATE == 0.0) & (Xdf.FOREIGN == 0.0)), 'OUT_OF_STATE_US'] = 1.0

    # Language
    if 'LANGUAGE_DESC' in Xdf_cols:
      Xdf[['ENGLISH', 'SPANISH', 'OTHER_LANGUAGE']] = 0.0
      Xdf.loc[(Xdf.LANGUAGE_DESC == 'English'), 'ENGLISH'] = 1.0
      Xdf.loc[(Xdf.LANGUAGE_DESC == 'Spanish'), 'SPANISH'] = 1.0
      Xdf.loc[(Xdf.LANGUAGE_DESC == 'Unable to Collect'), 'UNKNOWN_LANGUAGE'] = 1.0
      Xdf.loc[((Xdf.ENGLISH == 0.0) & (Xdf.SPANISH == 0.0) & (Xdf.UNKNOWN_LANGUAGE == 0.0)), 'OTHER_LANGUAGE'] = 1.0

      Xdf.drop(columns=['STATE_CODE', 'LANGUAGE_DESC'], inplace=True)
    return Xdf

  def join_with_all_deciles(self, Xdf: pd.DataFrame, col2decile_ftrs2aggf=None):
    """
    Join input dataframe with precomputed decile info

    :param Xdf: dataframe that contains the relevant medical columns that are *NOT* one-hot encoded!
    :param col2decile_ftrs2aggf:

    :return: a modified Xdf, added with all decile-related columns
    """
    if col2decile_ftrs2aggf is None:
      col2decile_ftrs2aggf = self.col2decile_ftr2aggf
    if col2decile_ftrs2aggf is None:
      return Xdf

    for col, ftr2aggf in col2decile_ftrs2aggf.items():
      if col == globals.PPROC:
        Xdf = self.join_with_pproc_decile(Xdf, self.decile_generator.pproc_decile, ftr2aggf)
      elif col == globals.CPT:
        Xdf = self.join_with_cpt_decile(Xdf, self.decile_generator.cpt_decile, ftr2aggf)
      elif col == globals.MED1:
        Xdf = self.join_with_med_decile(Xdf, self.decile_generator.med_level2decile[1], 1, ftr2aggf)
      elif col == globals.MED2:
        Xdf = self.join_with_med_decile(Xdf, self.decile_generator.med_level2decile[2], 2, ftr2aggf)
      elif col == globals.MED3:
        Xdf = self.join_with_med_decile(Xdf, self.decile_generator.med_level2decile[3], 3, ftr2aggf)
      else:
        raise NotImplementedError("%s decile is not implemented yet!" % col)
    return Xdf

  def join_with_cpt_decile(self, Xdf: pd.DataFrame, cpt_decile: pd.DataFrame,
                           decile_col2aggf={globals.CPT_DECILE: 'max'}):
    # Note: Assume Xdf contains the CPT list column before one-hot encoding
    # 1. Explode Xdf on column 'CPTS'; 2. Groupby 'SURG_CASE_KEY' and apply agg func
    Xdf_w_decile = Xdf[['SURG_CASE_KEY', 'CPTS']]\
      .explode('CPTS')\
      .dropna(subset=['CPTS'])\
      .rename(columns={'CPTS': 'CPT'})\
      .join(cpt_decile.set_index('CPT')[decile_col2aggf.keys()], on='CPT', how='inner')\
      .groupby('SURG_CASE_KEY')\
      .agg(decile_col2aggf)
    # Join on 'SURG_CASE_KEY' with the input Xdf
    Xdf_ret = Xdf.join(Xdf_w_decile, on='SURG_CASE_KEY', how='inner')
    print("[FtrEng-join_CPT_dcl] Input Xdf shape: ", Xdf.shape, '; Output Xdf shape: ', Xdf_ret.shape,
          "\nAdded decile columns: ", decile_col2aggf.keys())
    return Xdf_ret

  def join_with_med_decile(self, Xdf: pd.DataFrame, med_decile: pd.DataFrame, level=1,
                           decile_col2aggf={globals.MED1_DECILE: 'max'}):
    # Note: Assume Xdf contains the medication column (dtype is list) before one-hot encoding
    # 1. Explode Xdf on column 'LEVEL#_DRUG_CLASS_NAME'; 2. Groupby 'SURG_CASE_KEY' and apply agg func
    med_col = globals.DRUG_COLS[level-1]
    Xdf_w_decile = Xdf[['SURG_CASE_KEY', med_col]]\
      .explode(med_col)\
      .join(med_decile.set_index(med_col), on=med_col, how='left')\
      .groupby('SURG_CASE_KEY')\
      .agg(decile_col2aggf)\
      .fillna(0)  # TODO: carefully handle NaN here! Think about how different agg functions would handle NA
    # Join on 'SURG_CASE_KEY' with the input Xdf
    Xdf_ret = Xdf.join(Xdf_w_decile, on='SURG_CASE_KEY', how='inner')
    print("[FtrEng-join_MED%d_dcl] Input Xdf shape: " % level, Xdf.shape, '; Output Xdf shape: ', Xdf_ret.shape,
          "\nAdded decile columns: ", decile_col2aggf.keys())
    return Xdf_ret

  def join_with_pproc_decile(self, Xdf: pd.DataFrame, pproc_decile: pd.DataFrame,
                             decile_col2aggf={globals.PPROC_DECILE: 'max'}):
    Xdf_ret = Xdf.join(pproc_decile.set_index('PRIMARY_PROC')[decile_col2aggf.keys()], on='PRIMARY_PROC', how='inner')
    #  .groupby('SURG_CASE_KEY').agg(decile_col2aggf).reset_index() --- This part removes all other cols!
    print("[FtrEng-join_PPROC_decile] Input Xdf shape: ", Xdf.shape, "; Output Xdf shape: ", Xdf_ret.shape)
    if Xdf.shape[0] != Xdf_ret.shape[0]:
      raise Warning("%d cases are discarded due to unseen primary procedure!" % (Xdf.shape[0] - Xdf_ret.shape[0]))
    return Xdf_ret

  # Note: column order should be taken care of after calling this function
  def match_Xdf_cols_to_target_features(self, Xdf: pd.DataFrame, target_features):
    Xdf_cols = Xdf.columns.to_list()

    # Drop rows that has certain indicator columns not covered in the target feature list (e.g. an unseen CPT code)
    new_ftrs = set(Xdf_cols) - set(target_features) - set(globals.NON_NUMERIC_COLS)
    # TODO: think about how to handle different types of unseen codes (e.g. always drop if pproc is unseen, but unseen CCSR/CPT could be fine)
    if len(new_ftrs) > 0:
      case_idxs_with_new_ftrs = set()
      for new_ftr in new_ftrs:
        idxs = Xdf.index[Xdf[new_ftr] == 1].to_list()
        case_idxs_with_new_ftrs = case_idxs_with_new_ftrs.union(set(idxs))
      self._print_feature_match_details(new_ftrs, 'unseen')
      print("Dropping %d cases with new features..." % len(case_idxs_with_new_ftrs))
      Xdf = Xdf.drop(index=list(case_idxs_with_new_ftrs)) \
        .drop(columns=list(new_ftrs)) \
        .reset_index(drop=True)
      if Xdf.shape[0] == 0:
        raise Exception("All cases in this dataset contain at least 1 unseen indicator!")

    # Add unobserved indicators as columns of 0
    uncovered_ftrs = set(target_features) - set(Xdf_cols)
    Xdf[list(uncovered_ftrs)] = 0.0
    self._print_feature_match_details(uncovered_ftrs, 'uncovered')
    return Xdf

  def _print_feature_match_details(self, feature_set, ftr_type='unseen'):
    print("\nTotal %s features: %d" % (ftr_type, len(feature_set)))
    print("#%s CPTs: %d" % (ftr_type, len(list(filter(lambda x: x.startswith('CPT'), feature_set)))))
    print("#%s CPT Groups: %d" % (ftr_type, len(list(filter(lambda x: x.startswith('CPT_GROUP'), feature_set)))))
    print("#%s CCSRs: %d" % (ftr_type, len(list(filter(lambda x: x.startswith('CCSR'), feature_set)))))

  # Apply one-hot encoding to the designated columns
  def onehot_encode_cols(self, Xdf, onehot_cols, onehot_dtypes):
    if onehot_cols is None or onehot_dtypes is None:
      return Xdf

    for oh_col, dtype in zip(onehot_cols, onehot_dtypes):
      oh_prefix = oh_col if oh_col not in globals.DRUG_COLS else 'MED%s_' % list(filter(str.isdigit, oh_col))[0]
      if dtype == str:
        dummies = pd.get_dummies(Xdf[oh_col], prefix=oh_prefix)
      elif dtype == list:  # Expand list to (row_id, oh_col indicator) first
        s = Xdf[oh_col].explode()
        dummies = pd.crosstab(s.index, s).add_prefix(oh_prefix[:-1] + '_')
        dummies[dummies > 1] = 1  # in case a list contains duplicates  TODO: double check
      else:
        raise NotImplementedError("Cannot encode column '%s' with a data type of '%s'" % (oh_col, dtype))
      #Xdf = Xdf.drop(columns=[oh_col]).join(dummies).fillna(0)
      Xdf = Xdf.join(dummies).fillna(0)
    return Xdf

  def trim_ccsr_in_X(self, Xdf, onehot_cols, trimmed_ccsrs):
    if onehot_cols is None or trimmed_ccsrs is None: return Xdf, onehot_cols

    # add a column with only the target set of CCSRs
    if 'CCSRS' in onehot_cols:
      Xdf['Trimmed_CCSRS'] = Xdf['CCSRS'].apply(lambda row: [cc for cc in row if cc in trimmed_ccsrs])
      onehot_cols = list(map(lambda item: item.replace('CCSRS', 'Trimmed_CCSRS'), onehot_cols))
    # add a column with only the ICD10s of the target CCSR set
    if 'ICD10S' in onehot_cols:
      Xdf['Trimmed_ICD10S'] = Xdf[['CCSRS', 'ICD10S']].apply(
        lambda row: [row['ICD10S'][i] for i in range(len(row['ICD10S'])) if row['CCSRS'][i] in trimmed_ccsrs], axis=1)
      onehot_cols = list(map(lambda item: item.replace('ICD10S', 'Trimmed_ICD10S'), onehot_cols))
    return Xdf, onehot_cols

  def get_ccsr_decile(self):
    assert isinstance(self.decile_generator, DecileGenerator), \
      "Field 'decile_generator' must be a DecileGenerator object!"
    return self.decile_generator.ccsr_decile

  def get_cpt_decile(self):
    assert isinstance(self.decile_generator, DecileGenerator), \
      "Field 'decile_generator' must be a DecileGenerator object!"
    return self.decile_generator.cpt_decile

  def get_med_decile(self, level):
    assert isinstance(self.decile_generator, DecileGenerator), \
      "Field 'decile_generator' must be a DecileGenerator object!"
    return self.decile_generator.med_level2decile[level]

  def get_pproc_decile(self):
    assert isinstance(self.decile_generator, DecileGenerator), \
      "Field 'decile_generator' must be a DecileGenerator object!"
    return self.decile_generator.pproc_decile

  def set_decile_gen(self, decile_gen):
    assert isinstance(decile_gen, DecileGenerator), 'Input decile_gen must be a DecileGenerator object!'
    self.decile_generator = decile_gen


# Generator of medical complexity of different types of medical codes
class DecileGenerator(object):
  """
  Methods of this object should only be applied on Xtrain.
  For Xtest, simply join the test data frame with the corresponding decile df in this object
  """

  def __init__(self):
    self.pproc_decile = None
    self.cpt_decile = None
    self.cpt_group_decile = None
    self.ccsr_decile = None
    self.med_level2decile = {}

  def gen_decile_cols(self, Xdf: pd.DataFrame, outcome, col2decile_features=globals.DEFAULT_COL2DECILE_FTR2AGGF):
    """
    Generates the decile & related columns for each medical code type (CPT, CCSR, medication, etc.)

    :param Xdf: dataframe that contains the relevant columns that are *NOT* one-hot encoded!
    :param outcome: surgical outcome type to calculate decile score
    :param col2decile_features: A nested dict of medical code --> a dict of decile col name --> corresponding aggregation function

    :return: A list of decile-related features
    """
    if col2decile_features is None:
      print("No decile is generated.")
      return []

    X_dcl_features = []
    for col, dcl_ftrs2aggf in col2decile_features.items():
      if col == globals.PPROC:
        self.pproc_decile = self.gen_pproc_decile(Xdf, outcome)
      elif col == globals.CPT:
        self.cpt_decile = self.gen_cpt_decile(Xdf, outcome)
      elif col == globals.MED1:
        med1_decile = self.gen_medication_decile(Xdf, outcome, level=1)
        self.med_level2decile[1] = med1_decile
      elif col == globals.MED2:
        med2_decile = self.gen_medication_decile(Xdf, outcome, level=2)
        self.med_level2decile[2] = med2_decile
      elif col == globals.MED3:
        med3_decile = self.gen_medication_decile(Xdf, outcome, level=3)
        self.med_level2decile[3] = med3_decile
      else:
        raise NotImplementedError("%s decile is not implemented yet!" % col)
      X_dcl_features.extend(dcl_ftrs2aggf.keys())
    return X_dcl_features

  # Generate pproc sexile & other summary statistics
  def gen_pproc_decile(self, data_df, outcome=globals.LOS, save_fp=None):
    pproc_groupby = data_df[['PRIMARY_PROC', globals.LOS]].groupby(by=['PRIMARY_PROC'])

    pproc_decile = pproc_groupby[outcome].median().reset_index(name='PPROC_MEDIAN') \
      .join(pproc_groupby[outcome].mean().reset_index(name='PPROC_MEAN').set_index('PRIMARY_PROC'), on='PRIMARY_PROC', how='left') \
      .join(pproc_groupby.size().reset_index(name='PPROC_COUNT').set_index('PRIMARY_PROC'), on='PRIMARY_PROC', how='left') \
      .join(pproc_groupby[outcome].std().reset_index(name='PPROC_SD').set_index('PRIMARY_PROC'), on='PRIMARY_PROC', how='left') \
      .join(pproc_groupby[outcome].min().reset_index(name='PPROC_MIN').set_index('PRIMARY_PROC'), on='PRIMARY_PROC', how='left') \
      .join(pproc_groupby[outcome].max().reset_index(name='PPROC_MAX').set_index('PRIMARY_PROC'), on='PRIMARY_PROC', how='left')

    pproc_decile[globals.PPROC_DECILE] = pproc_decile['PPROC_MEDIAN'].apply(lambda x: float(min(round(x), globals.MAX_NNT + 1)))
    pproc_decile.reset_index(inplace=True)

    if save_fp:
      pproc_decile.to_csv(save_fp, index=False)
    return pproc_decile

  def gen_cpt_decile(self, data_df: pd.DataFrame, outcome=globals.LOS, save_fp=None):  # Only called for Xtrain
    cpt_df = data_df[['CPTS', outcome]]
    exp_cpt2outcome_df = cpt_df.explode('CPTS').dropna(subset=['CPTS']).rename(columns={'CPTS': 'CPT'})
    cpt_groupby = exp_cpt2outcome_df.groupby('CPT')
    cpt_decile = cpt_groupby[outcome].median().reset_index(name='CPT_MEDIAN').set_index('CPT') \
      .join(cpt_groupby[outcome].mean().reset_index(name='CPT_MEAN').set_index('CPT'), on='CPT', how='left') \
      .join(cpt_groupby.size().reset_index(name='CPT_COUNT').set_index('CPT'), on='CPT', how='left') \
      .join(cpt_groupby[outcome].std().reset_index(name='CPT_SD').set_index('CPT'), on='CPT', how='left') \
      .join(cpt_groupby[outcome].min().reset_index(name='CPT_MIN').set_index('CPT'), on='CPT', how='left') \
      .join(cpt_groupby[outcome].max().reset_index(name='CPT_MAX').set_index('CPT'), on='CPT', how='left')

    cpt_decile[globals.CPT_DECILE] = cpt_decile['CPT_MEDIAN'].apply(lambda x: float(min(round(x), globals.MAX_NNT + 1)))
    cpt_decile.reset_index(inplace=True)

    if save_fp:
      cpt_decile.to_csv(save_fp, index=False)
    return cpt_decile

  def gen_medication_decile(self, data_df: pd.DataFrame, outcome=globals.LOS, level=1, save_dir=None):
    med_col = globals.DRUG_COLS[level-1]
    med_outcome_df = data_df[[med_col, outcome]]
    # Explode df by medication type col and groupby
    exp_med_out_df = med_outcome_df.explode(med_col).dropna(subset=[med_col])
    med_groupby = exp_med_out_df.groupby(med_col)
    med_decile = med_groupby[outcome].median().reset_index(name='MED%d_Median' % level).set_index(med_col) \
      .join(med_groupby[outcome].mean().reset_index(name='MED%d_MEAN' % level).set_index(med_col), on=med_col, how='left') \
      .join(med_groupby.size().reset_index(name='MED%d_COUNT' % level).set_index(med_col), on=med_col, how='left')\
      .join(med_groupby[outcome].std().reset_index(name='MED%d_SD' % level).set_index(med_col), on=med_col, how='left')\
      .join(med_groupby[outcome].min().reset_index(name='MED%d_MIN' % level).set_index(med_col), on=med_col, how='left')\
      .join(med_groupby[outcome].max().reset_index(name='MED%d_MAX' % level).set_index(med_col), on=med_col, how='left')

    med_decile['MED%d_DECILE' % level] = med_decile['MED%d_Median' % level].apply(
      lambda x: float(min(round(x), globals.MAX_NNT + 1)))
    med_decile.reset_index(inplace=True)

    if save_dir:
      med_decile.to_csv(Path(save_dir) / ('med%d_decile.csv' % level), index=False)
    return med_decile
