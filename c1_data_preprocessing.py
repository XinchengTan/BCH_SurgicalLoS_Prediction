"""
Helper functions to preprocess the data and generate data matrix with its corresponding labels
"""
from collections import Counter

from IPython.display import display
from imblearn.over_sampling import SMOTENC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import globals
from . import c0_data_prepare as dp


class Dataset(object):

  def __init__(self, df, outcome=globals.NNT, cols=globals.FEATURE_COLS, onehot_cols=None, onehot_dtypes=None,
               test_pct=0.2, cohort=globals.COHORT_ALL, trimmed_ccsr=None, discretize_cols=None,
               denoise_what=None, denoise_how=None):
    if (onehot_cols is not None) and (onehot_dtypes is not None):
      assert len(onehot_cols) == len(onehot_dtypes), "One-hot Encoding columns and dtypes must match!"
    self.df = df
    self.cohort = cohort
    if cohort != globals.COHORT_ALL:  # Filter df by primary procedure cohort
      ch_pprocs = globals.COHORT_TO_PPROCS[cohort]
      self.cohort_df = df.query("PRIMARY_PROC in @ch_pprocs")
    else:
      self.cohort_df = df

    X, y, feature_cols, case_keys = gen_Xy(self.cohort_df, outcome, cols, onehot_cols, onehot_dtypes, trimmed_ccsr,
                                           discretize_cols)
    if test_pct > 0:
      self.Xtrain, self.Xtest, self.ytrain, self.ytest, self.train_idx, self.test_idx = gen_train_test(X, y, test_pct)
    else:
      self.Xtrain, self.ytrain, self.train_idx = X, y, np.arange(X.shape[0])
      self.Xtest, self.ytest, self.test_idx = np.array([]), np.array([]), np.array([])
    self.feature_names = feature_cols

    # TODO: keep the count of the original cohort df
    self.noisy_cases_df = None
    if (denoise_what != None) and (denoise_how != None):
      self.denoise(denoise_what, denoise_how)

    self.case_keys = case_keys
    self.sps_preds = df[globals.SPS_LOS_FTR].to_numpy()  # Might contain NaN
    self.cpt_groups = df['CPT_GROUPS'].to_numpy()

  def denoise(self, denoise_what, denoise_how):
    # TODO: Consider whether we need to update cohort df, given the set of rows are changed. -- I vote for no update for now

    if denoise_what == globals.DENOISE_ONLY_TRAIN:
      self._denoise_train(denoise_how)
    # elif denoise_what == globals.DENOISE_TRAIN_TEST0:  # denoise train, and then denoise test from noise in train
    #   self._denoise_test_from_train(denoise_how)
    # elif denoise_what == globals.DENOISE_TRAIN_TEST1:
    #   self._denoise_train(denoise_how)
    #   self._denoise_test_from_train(denoise_how)  # TODO: this might not be correct
    else:
      raise Warning("Skipping denoise because %s is not implemented yet!" % denoise_what)

  def _denoise_train(self, how):
    clean_Xytrain_df, cleaned_noisy_cases_df = denoise(self.Xtrain, self.ytrain, self.feature_names, how=how)
    self.Xtrain = clean_Xytrain_df[self.feature_names]
    self.ytrain = clean_Xytrain_df['Outcome']
    self.train_idx = clean_Xytrain_df.index
    self.noisy_cases_df = cleaned_noisy_cases_df

  def _denoise_test_from_train(self, how):
    if how == globals.DENOISE_ALL:
      # clean_Xytrain_df, o2m_noise = denoise(self.Xtrain, self.ytrain, self.feature_names, self.noisy_cases_df, globals.DENOISE_O2M)
      # TODO: double check correctness
      self._denoise_test_from_train(how=globals.DENOISE_O2M)
      o2m_noise_df = self.noisy_cases_df.copy()
      self._denoise_test_from_train(how=globals.DENOISE_PURE_DUP)
      self.noisy_cases_df = pd.concat([o2m_noise_df, self.noisy_cases_df])
    else:
      self._denoise_train(how)
      clean_Xytest_df, _ = denoise(self.Xtest, self.ytest, self.feature_names, self.noisy_cases_df, how)
      self.Xtest = clean_Xytest_df[self.feature_names]
      self.ytest = clean_Xytest_df['Outcome']
      self.test_idx = clean_Xytest_df.index
    # keep_idx_mask = np.ones_like(self.ytest, dtype=bool)
    # for i, row in enumerate(self.Xtest):
    #   for ns_idx, ns_row in self.noisy_cases_df.iterrows():
    #     if np.array_equal(ns_row[self.feature_names], row) and ns_row['Outcome'] != self.ytest[i]:
    #       keep_idx_mask[i] = False
    #
    # self._filter_X_by_index(globals.XTEST, keep_idx_mask)
    #
    # # TODO: this is DENOISE_TRAIN_TEST1, need 0
    # _, keep_idx = np.unique(self.Xtest, return_index=True, axis=0)
    # self._filter_X_by_index(globals.XTEST, keep_idx)

  def _filter_X_by_index(self, Xtype, keep_idxs):
    if Xtype == globals.XTEST:
      self.Xtest = self.Xtest[keep_idxs, :]
      self.ytest = self.ytest[keep_idxs]
      self.test_idx = self.test_idx[keep_idxs]
    else:
      raise NotImplementedError

  def __str__(self):
    res = "Training set size: %d\n" \
          "Test set size: %d\n"\
          "Number of features: %d\n\n" % (self.ytrain.shape[0], self.ytest.shape[0], len(self.feature_names))
    res += "Feature names: \n" + "\n".join(self.feature_names)
    return res


# TODO: add data cleaning, e.g. 1. check for negative LoS and remove; 2. check identical rows with different outcome
# Generate data matrix X
def gen_Xy(df, outcome=globals.NNT, cols=globals.FEATURE_COLS, onehot_cols=None, onehot_dtypes=None, trimmed_ccsr=None,
           discretize_cols=None):
  """
  Generate X, y for downstream modeling

  :param df:
  :param outcome:
  :param nranges: A list of non-negative numbers, starting with 0
  :param cols:
  :return:
  """
  # Make data matrix X
  X, feature_cols, X_case_key = gen_X(df, cols, onehot_cols, onehot_dtypes, trimmed_ccsr=trimmed_ccsr,
                                      discretize_cols=discretize_cols)

  # Get outcome vector
  y = gen_y(df, outcome)
  return X, y, feature_cols, X_case_key


def gen_X(df, cols=globals.FEATURE_COLS, onehot_cols=None, onehot_dtypes=None, remove_nonnumeric=True, verbose=False,
          trimmed_ccsr=None, discretize_cols=None):
  # Make data matrix X numeric
  X = df.copy()[cols]
  X.loc[(X.SEX_CODE != 'F'), 'SEX_CODE'] = 0.0
  X.loc[(X.SEX_CODE == 'F'), 'SEX_CODE'] = 1.0

  feature_cols = list(cols)
  if trimmed_ccsr:
    # add a column with only the target set of CCSRs
    if 'CCSRS' in onehot_cols:
      X['Trimmed_CCSRS'] = X['CCSRS'].apply(lambda row: [cc for cc in row if cc in trimmed_ccsr])
      onehot_cols = list(map(lambda item: item.replace('CCSRS', 'Trimmed_CCSRS'), onehot_cols))
      feature_cols = list(map(lambda item: item.replace('CCSRS', 'Trimmed_CCSRS'), feature_cols))
    # add a column with only the ICD10s of the target CCSR set
    if 'ICD10S' in onehot_cols:
      X['Trimmed_ICD10S'] = X[['CCSRS', 'ICD10S']].apply(lambda row: [row['ICD10S'][i]
                                                                      for i in range(len(row['ICD10S']))
                                                                      if row['CCSRS'][i] in trimmed_ccsr], axis=1)
      onehot_cols = list(map(lambda item: item.replace('ICD10S', 'Trimmed_ICD10S'), onehot_cols))
      feature_cols = list(map(lambda item: item.replace('ICD10S', 'Trimmed_ICD10S'), feature_cols))

  if onehot_cols is not None:
    # Apply one-hot encoding to the designated columns
    for oh_col, dtype in zip(onehot_cols, onehot_dtypes):
      if dtype == str:  # can directly use get_dummies()
        dummies = pd.get_dummies(X[oh_col], prefix=oh_col)
      elif dtype == list:  # Need to expand list to (row_id, oh_col indicator) first
        s = X[oh_col].explode()
        dummies = pd.crosstab(s.index, s).add_prefix(oh_col[:-1] + '_')
        dummies[dummies > 1] = 1  # in case a list contains duplicates
        # # Alternative:
        # dummies = X[oh_col].apply(lambda x: pd.Series(1, x))
        # X = pd.concat([X.drop(columns=[oh_col]), dummies.fillna(0)], axis=1)
      else:
        raise NotImplementedError
      X = X.drop(columns=[oh_col]).join(dummies).fillna(0)
      feature_cols.remove(oh_col)
      feature_cols.extend(dummies.columns.to_list())

  # Save SURG_CASE_KEY, but drop with other non-numeric columns for data matrix
  X_case_key = X['SURG_CASE_KEY'].to_numpy()
  if remove_nonnumeric:
    X.drop(columns=globals.NON_NUMERIC_COLS, inplace=True, errors='ignore')  #
    for nnm_col in globals.NON_NUMERIC_COLS:
      if nnm_col in feature_cols:
        feature_cols.remove(nnm_col)

  # Bucket SPS predicted LoS into 9 classes, if there such prediction exists
  if globals.SPS_LOS_FTR in cols:  # Assume SPS prediction are all integers?
    X.loc[(X[globals.SPS_LOS_FTR] > globals.MAX_NNT), globals.SPS_LOS_FTR] = globals.MAX_NNT + 1

  # Discretize certain continuous columns by request
  X = X.to_numpy(dtype=np.float64)
  if discretize_cols:
    discretize_columns(X, feature_cols, discretize_cols, inplace=True)

  if verbose:
    display(pd.DataFrame(X, columns=feature_cols).head(20))

  # Basic sanity check
  assert X.shape[1] == len(feature_cols), 'Generated data matrix has %d features, but feature list has %d items' % \
                                          (X.shape[1], len(feature_cols))
  assert len(set(feature_cols)) == len(feature_cols), "Generated data matrix contains duplicated feature names!"

  return X, feature_cols, X_case_key


def discretize_columns(X, feature_names, discretize_cols, inplace=False):
  if not inplace:
    X = np.copy(X)

  # Modify data matrix with discretized columns by request
  for dis_col in discretize_cols:
    idx = feature_names.index(dis_col)
    if dis_col == 'AGE_AT_PROC_YRS':
      X[:, idx] = np.digitize(X[:, idx], globals.AGE_BINS)
    elif dis_col == 'WEIGHT_ZSCORE':
      weightz_bins = [float('-inf'), -4, -2, -1, 1, 2, 4, float('inf')]
      print("Weight z-score bins: ", weightz_bins)
      X[:, idx] = np.digitize(X[:, idx], weightz_bins)
    else:
      raise Warning("%s discretization is not available yet!" % dis_col)
  return X


def gen_y(df, outcome):
  # Generate an outcome vector y with shape: (n_samples, )
  y = np.array(df.LENGTH_OF_STAY.to_numpy())
  if outcome == globals.LOS:
    return y
  elif outcome == ">12h":
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
  elif outcome == ">1d":
    y[y > 1] = 1
    y[y <= 1] = 0
  elif outcome == globals.NNT:
    y = gen_y_nnt(df[globals.NNT])
  elif outcome.endswith("nnt"):
    cutoff = int(outcome.split("nnt")[0])
    y = gen_y_nnt_binary(y, cutoff)
  else:
    raise NotImplementedError("Outcome type '%s' is not implemented yet!" % outcome)
  return y


def gen_y_nnt(dfcol):
  y = np.array(dfcol.to_numpy())
  y[y > globals.MAX_NNT] = globals.MAX_NNT + 1
  return y


def gen_y_nnt_binary(y, cutoff):  # task: predict if LoS <= cutoff (cutoff in range(0, 8)
  yb = np.copy(y)
  yb[y <= cutoff] = 1
  yb[y > cutoff] = 0
  return yb


# Perform train-test split
def gen_train_test(X, y, test_pct=0.2):
  """
  Returns the purely integer-location based index w.r.t. the data matrix X for the train and test set respectively.
  i.e. max(X_train.index) <= n_samples - 1 and max(X_test.index) <= n_samples - 1
  These index lists are not the row index label of the original df, because X is already a numeric matrix and
  has lost the index labels during the conversion.

  :param X: a numeric matrix (n_samples, n_features)
  :param y: a response vector (n_samples, )
  :param test_pct: desired test set percentage, a float between 0 and 1

  :return: training and test set data matrix, response vector and location-based index w.r.t. X
  """
  X, y = pd.DataFrame(X), pd.Series(y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, random_state=globals.SEED)
  # print("[gen_train_test] X index: ", X.index)
  # print("min index, max index: ", min(X.index), max(X.index))
  # print("train - min, max: ", min(X_train.index), max(X_train.index))
  # print("test - min, max: ", min(X_test.index), max(X_test.index))
  return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), X_train.index.to_numpy(), X_test.index.to_numpy()


# train-validation-test split
def gen_train_val_test(X, y, val_pct=0.2, test_pct=0.2):
  X, y = pd.DataFrame(X), pd.Series(y)
  X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_pct, random_state=globals.SEED)
  X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_pct, random_state=globals.SEED)
  return X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy(),\
         X_train.index, X_val.index, X_test.index


def standardize(X_train, X_test=None):
  scaler = StandardScaler().fit(X_train)
  if X_test is None:
    return scaler.transform(X_train)
  return scaler.transform(X_test)


def gen_smote_Xy(X, y, feature_names):
  categ_ftrs = np.array([False if ftr in globals.CONTINUOUS_COLS else True for ftr in feature_names], dtype=bool)
  sm = SMOTENC(categorical_features=categ_ftrs, random_state=globals.SEED)
  X, y = sm.fit_resample(X, y)
  return X, y


def denoise(X, y, features, how=globals.DENOISE_DEL_O2M):
  """
  Denoise dataset (X, y) by removing pure duplicates,
  or coalescing one-to-many cases to one case with its majority outcome,
  or doing both operations.

  :param X:
  :param y:
  :param noisy_cases: If not None, denoise these cases from X
  :param how:
  :return: A dataframe of X, y, with the index of the original X preserved
  """
  # Make Xydf_dup and Xydf_nodup
  Xydf = pd.DataFrame(X, columns=features)
  Xydf['Outcome'] = y
  dup_mask = Xydf.duplicated(subset=features, keep=False)
  Xydf_dup = Xydf[dup_mask]

  Xydf_clean = Xydf.drop_duplicates(subset=features, keep=False)

  o2m_keep_df = remove_o2m_cases(Xydf_dup, features)
  pure_dup_keep_df = remove_pure_dups(Xydf_dup, features)

  if how == globals.DENOISE_ALL:
    cleaned_cases_df = pd.concat([o2m_keep_df, pure_dup_keep_df])
    Xydf_clean = pd.concat([Xydf_clean, cleaned_cases_df])
  elif how == globals.DENOISE_O2M:
    cleaned_cases_df = o2m_keep_df
    #print("\n pure dup df: ", pure_dup_keep_df.columns)
    #print("\n Xydf: ", Xydf_dup.columns)
    kept_noise_df = pd.merge(Xydf_dup, pure_dup_keep_df, on=features, how='left', indicator=True, suffixes=('', '_r'))\
      .loc[lambda x: x['_merge'] != 'both']
    #print("\n kept noise df: ", kept_noise_df.columns)
    Xydf_clean = pd.concat([Xydf_clean, cleaned_cases_df, kept_noise_df[features + ['Outcome']]])
  elif how == globals.DENOISE_PURE_DUP:
    cleaned_cases_df = pure_dup_keep_df
    kept_noise_df = pd.merge(Xydf_dup, o2m_keep_df, on=features, how='left', indicator=True, suffixes=('', '_r'))\
      .loc[lambda x: x['_merge'] != 'both']
    Xydf_clean = pd.concat([Xydf_clean, cleaned_cases_df, kept_noise_df[features + ['Outcome']]])

  else:
    raise NotImplementedError

  assert len(Xydf_clean.columns) == len(features) + 1, "Clean Xy df has wrong number of columns!"
  #   assert True not in Xydf_clean.duplicated(subset=features), "Clean Xy df still has cases with duplications!"

  return Xydf_clean, cleaned_cases_df


def remove_o2m_cases(Xydup_df, features, noisy_cases_df=None):
  # o2m_df = Xydup_df.groupby(by=features)\
  #   .filter(lambda x: len(x.value_counts().index) > 1)
  #   .groupby(by=features)['Outcome'].count().sort_values('pos').groupby(level=len(features)-1).tail(1)
  Xydup_df = Xydup_df.groupby(by=features).filter(lambda x: len(x.value_counts().index) > 1)  # o2m cases
  o2m_grp2idxs = Xydup_df.groupby(by=features).groups
  o2m_keep_X, y, indices = [], [], []
  if not isinstance(noisy_cases_df, pd.DataFrame):  # noisy_cases_df == None or its equivalent
    for ftr, idxs in o2m_grp2idxs.items():
      outcomes = Xydup_df.loc[idxs]['Outcome'].to_list()
      outcome_counter = Counter(outcomes)
      if len(outcome_counter) > 1:
        o2m_keep_X.append(ftr)
        cur_y = max(outcome_counter, key=outcome_counter.get)
        y.append(cur_y)
        indices.append(idxs[outcomes.index(cur_y)])
    o2m_keep_df = pd.DataFrame(o2m_keep_X, columns=features, index=indices)
    o2m_keep_df['Outcome'] = y
    return o2m_keep_df
  else:
    for ftr, idxs in o2m_grp2idxs.items():
      noisy_case_match = noisy_cases_df[(noisy_cases_df[features] == ftr).all(1)]
      if len(noisy_case_match) > 0:
        selected_y = noisy_case_match.iloc[0]['Outcome']
        ftr_matched_cases = Xydup_df.loc[idxs]
        outcome_dismatch_cases = ftr_matched_cases[ftr_matched_cases['Outcome'] != selected_y]
        Xydup_df.drop(index=outcome_dismatch_cases.index.to_list(), inplace=True)
    Xydup_df.drop_duplicates(keep='first', inplace=True)
    return Xydup_df

      # if len(noisy_case_match) > 0:
      #   outcomes = Xydup_df.loc[idxs]['Outcome'].to_list()
      #   selected_y = noisy_case_match.iloc[0]['Outcome']
      #   if selected_y in outcomes:
      #     o2m_keep_X.append(ftr)
      #     y.append(selected_y)
      #     indices.append(idxs[outcomes.index(selected_y)])


def remove_pure_dups(Xydup_df, features, noisy_cases_df=None):
  if not isinstance(noisy_cases_df, pd.DataFrame):  # noisy_cases_df == None
    pure_dup_keep_df = Xydup_df.groupby(by=features)\
      .filter(lambda x: len(x.value_counts().index) == 1)\
      .drop_duplicates(subset=features, keep='first')
  else:
    pure_dup_keep_df = Xydup_df.groupby(by=features) \
      .filter(lambda x: len(x.value_counts().index) == 1)



    join_df = pd.merge(pure_dup_keep_df, noisy_cases_df, on=features, how='left', suffixes=('_l', '_r'), indicator=True)
    pure_dup_keep_df = join_df[join_df['_merge'] == 'left_only'].drop('_merge', axis=1, inplace=True)
    display(pure_dup_keep_df.head(5))
    overlap = join_df[join_df['_merge'] == 'both']\
      .drop_duplicates(subset=features+['Outcome_l'], keep='first')\
      .rename(columns={'Outcome_l': 'Outcome'}, inplace=True)\
      .drop(['Outcome_r', '_merge'], axis=1, inplace=True)
    pure_dup_keep_df = pd.concat([pure_dup_keep_df, overlap])

  return pure_dup_keep_df

