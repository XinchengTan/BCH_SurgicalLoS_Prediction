from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .. import globals


def pproc_cohort_ccsr_eda(ppcc_df, topK_ccsr=10, cohort=globals.COHORT_TO_PPROCS):
  """

  :param ppcc_df: Output df of 'primary_proc_assoc_ccsr_eda()'
  :param cohort: A mapping from a procedure group to a set of primary procedure name
  :return:
  """
  # Pick the top K most frequent CCSR for each pproc
  cohort2df = dict()
  for cohort, pprocs in cohort.items():
    ch_df = ppcc_df.query("`Primary Procedure` in @pprocs")

    cohort2df[cohort] = ch_df
  return cohort2df


def primary_proc_assoc_ccsr_eda(df, topK_ccsr=10):
  """
  Generate a table of primary procedure and a list of its CCSRs sorted by frequency descendingly
  Assume df is a dataframe to be preprocessed
  """
  N = df.shape[0]
  ccsrs = list(df.filter(regex='^CCSR_', axis=1).columns)
  pprocs = list(df['PRIMARY_PROC'].unique())
  pproc2stats = gen_pproc_stats_dict(df, N)

  pproc2ccsr_nnts = defaultdict(lambda: defaultdict(list))
  pproc2count = defaultdict(int)
  pproc2ccsr_stats = defaultdict(lambda: defaultdict(lambda: {}))

  # Groupby a combination of primary procedure and ccsr
  for i in tqdm(range(N), 'Generating pproc & ccsr - nnt list'):
    row = df.iloc[i]
    for ccsr_col in ccsrs:
      if row[ccsr_col] == 1:
        pproc2ccsr_nnts[row['PRIMARY_PROC']][ccsr_col].append(row[globals.NNT])

  # Compute aggregated stats for each pproc & ccsr combination
  for pproc in tqdm(pprocs, 'Generating agg stats for each pproc & ccsr'):
    for ccsr in ccsrs:
      cur_nnts = pproc2ccsr_nnts[pproc][ccsr]
      if len(cur_nnts) != 0:
        pproc2count[pproc] += len(cur_nnts)
        pproc2ccsr_stats[pproc][ccsr] = gen_stats_dict(cur_nnts, N)
    pproc2ccsr_stats[pproc]["Procedure Total"] = pproc2stats[pproc]

  # First, sort by pproc frequency descendingly, then within each pproc, sort by ccsr frequency descendingly
  pproc2count = dict(sorted(pproc2count.items(), key=lambda item: item[1], reverse=True))
  pproc2ccsr_stats = {pproc: dict(sorted(pproc2ccsr_stats[pproc].items(), key=lambda item: item[1]['Count'], reverse=True))
                      for pproc in pproc2count}

  # Convert the nested dict to tuple-keyed 1D dict and then to pd.Dataframe
  ppcc2stats = {(pp, cc): pproc2ccsr_stats[pp][cc]
                for pp in pproc2ccsr_stats.keys()
                for cc in list(pproc2ccsr_stats[pp].keys())[:topK_ccsr+1]}  # Pick the top K most frequent CCSR for each pproc
  ret_df = pd.DataFrame.from_dict(ppcc2stats, orient='index')
  ret_df.index.set_names(['Primary Procedure', 'CCSR'], inplace=True)

  return ret_df, pproc2count, pproc2ccsr_stats, pproc2ccsr_nnts

  # 1. select a CCSR-prefix subset of cols
  # 2. groupby pproc
  # 3. aggregate each column by count
  # df = pd.concat([df['PRIMARY_PROC'], df[globals.NNT],
  #                df.filter(regex='^CCSR_', axis=1).copy(deep=True)],
  #                axis=1)
  # ccsr_cols = None  # TODO: filter a list by prefix
  # ret = df.groupby('PRIMARY_PROC')[ccsr_cols].count()\
  #   .reset_index(name='frequency')


def gen_pproc_stats_dict(df, N):
  pproc2stats = df.groupby(by=['PRIMARY_PROC']).size()\
    .reset_index(name='Count')\
    .set_index('PRIMARY_PROC')
  pproc2stats['Count (%)'] = pproc2stats['Count'].apply(lambda x: "{:.2%}".format(x / N))
  pproc2stats_df = pproc2stats.join(
    df.groupby(by=['PRIMARY_PROC'])[globals.NNT]
      .mean()
      .reset_index(name='Mean')
      .set_index('PRIMARY_PROC'),
    on='PRIMARY_PROC', how='left'
  ).join(
    df.groupby(by=['PRIMARY_PROC'])[globals.NNT]
      .median()
      .reset_index(name='Median')
      .set_index('PRIMARY_PROC'),
    on='PRIMARY_PROC', how='left'
  ).join(
    df.groupby(by=['PRIMARY_PROC'])[globals.NNT]
      .std()
      .reset_index(name='NNT Std')
      .set_index('PRIMARY_PROC'),
    on='PRIMARY_PROC', how='left'
  )

  return pproc2stats_df.to_dict(orient='index')


def gen_stats_dict(nnts, N):
  return {'Count': len(nnts),
          'Count (%)': "{:.2%}".format(len(nnts) / N),
          'Mean': np.mean(nnts),
          'Median': np.median(nnts),
          'NNT Std': np.std(nnts)
          }