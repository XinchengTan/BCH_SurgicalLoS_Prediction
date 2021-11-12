from collections import defaultdict
import pandas as pd
from . import globals

# Drop outliers (i.e. patient who has an extremely long LOS)
def drop_outliers(dashboard_df, exclude_outliers=0, inplace=True):
  if exclude_outliers > 0:
    print("Initial number of rows", dashboard_df.shape[0])
    if exclude_outliers == 0:
      return dashboard_df
    else:
      if inplace:
        dashboard_df.drop(index=dashboard_df.nlargest(exclude_outliers, 'LENGTH_OF_STAY').index, inplace=True)
        ret_df = dashboard_df
      else:
        ret_df = dashboard_df.drop(index=dashboard_df.nlargest(exclude_outliers, 'LENGTH_OF_STAY').index)
      print("After dropping outliers (number of rows)", ret_df.shape[0])
      return ret_df


def gen_md2AllClfs(md2reg=None, md2multiclf=None, md2binclfs=None):
  md2allclfs = defaultdict(lambda : defaultdict(list))
  if md2reg is not None:
    for md, clf in md2reg.items():
      md2allclfs[md][globals.REG].append(clf)
  if md2multiclf is not None:
    for md, clf in md2multiclf.items():
      md2allclfs[md][globals.MULTI_CLF].append(clf)
  if md2binclfs is not None:
    for md, cutoff2clf in md2binclfs.items():
      md2allclfs[md][globals.BIN_CLF] = [cutoff2clf[c] for c in globals.NNT_CUTOFFS]

  return md2allclfs
