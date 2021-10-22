from collections import defaultdict

import numpy as np
import pandas as pd


def gen_cpt_groups(cpt_df, granularity=7, verbose=True):
  """
  Build a mapping between CPT category name and a list of CPT codes.
  The most detailed group is from bucket_'granularity'. CPTs with Nan category
  at each level above are assigned to its lowest-level category.

  :param cpt_df:
  :param granularity: an int between 1 and 7 (TODO: This is a key param to watch in EDA)

  :return: a mapping from CPT category to a list of CPTs
  """
  assert 1 <= granularity <= 7, "granularity is an int in [1, 7]"
  cpt_init = cpt_df[cpt_df['bucket_%d'%granularity].notnull()].groupby(['bucket_%d'%granularity])
  cpt_groups = {k: list(v) for k, v in cpt_init.groups.items()}  # value: a list of CPT code indices = d
  cpt_groups = defaultdict(list, cpt_groups)
  if granularity == 1:
    return cpt_groups

  dup_cpt_grp = defaultdict(list)  # a mapping of CPT group name to a list of bucket levels that it occurs
  for i in range(granularity-1, 0, -1):
    if i > 1:
      cpt_i = cpt_df[(cpt_df['bucket_%d'%(i + 1)].isnull()) & (cpt_df['bucket_%d'%i].notnull())].groupby(['bucket_%d'%i])
    else:
      # add cpts with only bucket 1 to the singleton groups (bucket_1 is never null)
      cpt_i = cpt_df[cpt_df['bucket_2'].isnull()].groupby(['cpt_description'])

    # update dictionaries
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
        for df_idx in v:
          actual_k = cpt_df.iloc[df_idx][i]
          cpt_groups[actual_k].append(df_idx)
    if verbose:
      print("Level %d: #cpts covered in grouping: " % i, sum(len(v) for v in cpt_groups.values()), "\n")

  if verbose:
    print("Duplicated CPT groups and their occurrence bucket levels:")
    correct_dup2levels = dict()
    for k, levels in dup_cpt_grp.items():
      complete_levels = list(levels)
      for l in levels:
        for i in range(l, 8):
          if k in cpt_df['bucket_%d'%i].unique() and i not in levels:
            complete_levels.append(i)
      # If len == 1, it's actually not a duplicate
      if len(complete_levels) > 1:
        print(k, complete_levels)
        correct_dup2levels[k] = complete_levels

  return cpt_groups, correct_dup2levels


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
        raise ValueError("Found code not in CPT dataframe! %d" % c)
      idx = all_cpt_codes.index(c)
      cpt_markers[idx] = 1 - cpt_markers[idx]  # flip the bit

  if np.sum(cpt_markers) != len(all_cpt_codes):
    raise ValueError("Some CPT code is not covered by the grouping!")
  return True


def gen_cpt_decile(df, score_range, by_code=True):
  """
  Generates CPT deciles based on CPT code or CPT group.
  TODO: note that score_range is a tunable parameter

  :param df:
  :param by_code: if true, generate decile based on each CPT code; otherwise, base on CPT grouping
  :param score_range: range of the decile score, starting from 1
  :return: a dictionary mapping each decile feature to a score
  """

  return


