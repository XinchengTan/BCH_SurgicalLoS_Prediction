from collections import defaultdict

import numpy as np
import pandas as pd
from globals_fs import *


def pad_cpt_map(cpt_df: pd.DataFrame, max_level=4, save_maxlevel_mapping=False):
  assert np.all(cpt_df['bucket_1'].notnull()), 'Bucket-1 CPT Group cannot contain NA!'
  assert max_level < 8, 'max_level is at most 7!'
  cpt_df_padded = cpt_df.copy()
  for level in range(2, max_level+1):
    cur_level_col = f'bucket_{level}'
    cpt_df_padded.loc[cpt_df_padded[cur_level_col].isna(), cur_level_col] = cpt_df_padded[f'bucket_{level-1}']
    cpt_df_padded.loc[cpt_df_padded[cur_level_col] == 'Other Procedures', cur_level_col] = cpt_df_padded[f'bucket_{level-1}']

  if save_maxlevel_mapping:
    df = cpt_df_padded[['cpt_code', f'bucket_{max_level}']].rename(columns={'cpt_code': 'CPT_CODE',
                                                                            f'bucket_{max_level}': 'CPT_GROUP'})
    df.to_csv(DEPLOY_DEP_FILES_DIR / f'cpt2group_level{max_level}.csv', index=False)
  return cpt_df_padded





# --------------------------------- IGNORE THE FOLLOWING CODE! GOT WEIRD BUG SOMEHOW... --------------------------------
def gen_cpt_groups(cpt_df, max_level=7, verbose=True):
  """
  Build a mapping between CPT category name and a list of CPT codes.
  The most detailed group is from bucket_'max_level'. CPTs with Nan category
  at each level above are assigned to its lowest-level category.

  :param cpt_df:
  :param max_level: an int between 1 and 7 (TODO: This is a key param to watch in EDA)

  :return: a mapping from CPT category to a list of CPTs
  """
  assert 1 <= max_level <= 7, "max_level is an int in [1, 7]"
  cpt_init = cpt_df[cpt_df[f'bucket_{max_level}'].notnull()].groupby([f'bucket_{max_level}'])
  cpt_groups = {k: list(v) for k, v in cpt_init.groups.items()}  # value: a list of CPT code indices = d
  cpt_groups = defaultdict(list, cpt_groups)
  if max_level == 1:
    return cpt_groups
  if verbose:
    print(f"Level {max_level}: #cpts covered in grouping: ", sum(len(v) for v in cpt_groups.values()))
    # print("Number of CPTs with level %d category: %d" % (i, CPT_TOTAL - cpt_map['bucket_%d' % i].isnull().sum()))
    print("Number of unique CPT categories: %d\n" % cpt_df[f'bucket_{max_level}'].nunique(dropna=True), "\n")

  # Iterate bottom up, starting from max_level initialized in cpt_groups
  dup_cpt_grp = defaultdict(list)  # a mapping of CPT group name to a list of bucket levels that it occurs
  for i in range(max_level-1, 0, -1):
    print('Level ', i)
    if i > 1:
      cpt_i = cpt_df[(cpt_df[f'bucket_{i+1}'].isnull()) & (cpt_df[f'bucket_{i}'].notnull())].groupby([f'bucket_{i}'])
    else:
      # add cpts with only bucket 1 to the singleton groups (bucket_1 is never null)
      cpt_i = cpt_df[cpt_df['bucket_2'].isnull()].groupby(['cpt_description'])

    # update dictionaries
    # groupby.groups.items: [(group_name, Int64Index([cpt_df_idx0, cpt_df_idx1, ... ]))]
    for k, v in cpt_i.groups.items():
      if k != 'Other Procedures':
        if k not in cpt_groups.keys():
          cpt_groups[k] = list(v)
        else:
          cpt_groups[k] = list(cpt_groups[k])
          cpt_groups[k].extend(list(v))
          dup_cpt_grp[k].append(i)  # Might falsely find a duplicate whose key is from the previous bucket of 'Other Procedures'
      else:
        # assign all CPT indices to the (i-1)th bucket
        for cpt_df_idx in v:
          prev_level_k = cpt_df.iloc[cpt_df_idx][i]
          cpt_groups[prev_level_k].append(cpt_df_idx)
    if verbose:
      print("Level %d: #cpts covered in grouping: " % i, sum(len(v) for v in cpt_groups.values()))
      #print("Number of CPTs with level %d category: %d" % (i, CPT_TOTAL - cpt_map['bucket_%d' % i].isnull().sum()))
      print("Number of unique CPT categories: %d\n" % cpt_df[f'bucket_{i}'].nunique(dropna=True), "\n")

  if verbose:
    print("Duplicated CPT groups and their occurrence bucket levels:")
    correct_dup2levels = dict()
    for k, levels in dup_cpt_grp.items():
      complete_levels = list(levels)
      for l in levels:
        for i in range(l, 8):
          if k in cpt_df[f'bucket_{i}'].unique() and i not in levels:
            complete_levels.append(i)
      # If len == 1, it's actually not a duplicate
      if len(complete_levels) > 1:
        print(k, complete_levels)
        correct_dup2levels[k] = complete_levels
    return cpt_groups, correct_dup2levels

  return cpt_groups


def gen_groupCode_mappings(cpt_df, cpt_groups):
  group2cpts = defaultdict(list)
  for k, v in cpt_groups.items():
    for idx in v:
      group2cpts[k].append(cpt_df.iloc[idx].cpt_code)

  cpt2group = dict()
  for k, v in group2cpts.items():
    for c in v:
      cpt2group[c] = k

  return group2cpts, cpt2group


def gen_cpt2group_df(cpt2group, path='./Data/cpt2group.csv'):
  df = pd.DataFrame(list(cpt2group.items()), columns=['CPT_CODE', 'CPT_GROUP'])
  return df.to_csv(path, index=False)


def is_valid_cpt_grouping(cpt_df, group2cpts):
  ":returns True if all CPTs covered by the grouping is exactly the entire CPT set."
  all_cpt_codes = cpt_df.cpt_code.tolist()

  cpt_markers = np.zeros(len(all_cpt_codes))
  for k, v in group2cpts.items():
    for c in v:
      if c not in all_cpt_codes:
        raise ValueError(f"Found code not in CPT dataframe! {c}")
      idx = all_cpt_codes.index(c)
      cpt_markers[idx] = 1 - cpt_markers[idx]  # flip the bit

  if np.sum(cpt_markers) != len(all_cpt_codes):
    raise ValueError("Some CPT code is not covered by the grouping!")
  return True
