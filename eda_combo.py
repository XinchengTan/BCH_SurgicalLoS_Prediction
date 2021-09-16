"""
EDA on diagnosis code, in particular grouping of these diagnosis codes
"""

from collections import defaultdict
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from globals import diaglabels
import plot_utils


def gen_cooccurrence_graph(dashboard_df, diagcode_type='OS', threshold=500):
  # Plot a connectivity graph of pairwise cooccurrence (darker edge color means higher cooccurrences)
  A, _ = gen_cooccurrence_matrix(dashboard_df, diagcode_type=diagcode_type)
  plot_utils.plot_connectivity_graph(A, threshold=threshold)


def cooccurrence_los_eda(dashboard_df, diagcode_type='OS', order=2, K=30):
  combo_freq, combo_aggStats = gen_cooccurrence_matrix(dashboard_df, order, diagcode_type)
  topK_combo_df = gen_topK_frequent_combo_with_losStats(combo_freq, combo_aggStats, K=K, order=order, diagcode=diagcode_type)
  return topK_combo_df


def gen_cooccurrence_matrix(dashboard_df, order=2, diagcode_type='OS'):
  D = len(diaglabels)

  if order == 2 and diagcode_type == 'OS':
    diag_pair_freq = np.zeros((D, D))
    aggStats_os2 = defaultdict(lambda x: [0, 0, 0])  # [mean, median, std]
    for i in range(D):
      for j in range(i + 1, D):
        di, dj = diaglabels[i], diaglabels[j]
        ij_los = dashboard_df[(dashboard_df[di] == 1) & (dashboard_df[dj] == 1)].LENGTH_OF_STAY
        diag_pair_freq[i][j] = ij_los.shape[0]
        if ij_los.shape[0] > 0:
          aggStats_os2[(i, j)] = [np.mean(ij_los), np.median(ij_los), np.std(ij_los)]
    return diag_pair_freq, aggStats_os2

  elif order == 3 and diagcode_type == 'OS':
    os3_freq = np.zeros((D, D, D))
    aggStats_os3 = np.zeros((D, D, D, 3))
    for i in range(D):
      for j in range(i + 1, D):
        for k in range(j + 1, D):
          di, dj, dk = diaglabels[i], diaglabels[j], diaglabels[k]
          ijk_los = dashboard_df[
            (dashboard_df[di] == 1) & (dashboard_df[dj] == 1) & (dashboard_df[dk] == 1)].LENGTH_OF_STAY
          os3_freq[i][j][k] = ijk_los.shape[0]
          if ijk_los.shape[0] > 0:
            aggStats_os3[i][j][k] = np.array([np.mean(ijk_los), np.median(ijk_los), np.std(ijk_los)])
    return os3_freq, aggStats_os3

  elif order == 4 and diagcode_type == 'OS':
    os4_freq = defaultdict(int)
    aggStats_os4 = defaultdict(lambda x: [0, 0, 0])
    for i in range(D):
      for j in range(i + 1, D):
        for k in range(j + 1, D):
          for s in range(k + 1, D):
            di, dj, dk, ds = diaglabels[i], diaglabels[j], diaglabels[k], diaglabels[s]
            ijks_los = dashboard_df[(dashboard_df[di] == 1) & (dashboard_df[dj] == 1) & (dashboard_df[dk] == 1) & (
                  dashboard_df[ds] == 1)].LENGTH_OF_STAY
            os4_freq[(i, j, k, s)] = ijks_los.shape[0]
            if ijks_los.shape[0] > 0:
              aggStats_os4[(i, j, k, s)] = [np.mean(ijks_los), np.median(ijks_los), np.std(ijks_los)]
    return os4_freq, aggStats_os4


def gen_topK_frequent_combo_with_losStats(freq_df, aggStats_df, K=30, order=2, diagcode='OS'):
  if diagcode == 'OS':
    if order == 2:
      sorted_idxs = np.dstack(np.unravel_index(np.argsort(freq_df.ravel()), np.shape(freq_df)))[0]
      topK_idxs = np.flip(sorted_idxs[-K:], axis=0)  # K x 2

      topK_os2 = [[diaglabels[i], diaglabels[j], freq_df[i, j]] + aggStats_df[(i, j)] for i, j in topK_idxs]
      topK_os2_df = pd.DataFrame(topK_os2, columns=['OS code1', 'OS code2', 'Cooccurrence_Count',
                                                    'Mean', 'Median', 'Std'])
      return topK_os2_df

    if order == 3:
      sorted_idxs = np.dstack(np.unravel_index(np.argsort(freq_df.ravel()), np.shape(freq_df)))[0]
      topK_idxs = np.flip(sorted_idxs[-K:], axis=0)  # K x 3
      topK_os3 = [[diaglabels[i], diaglabels[j], diaglabels[k], freq_df[i, j, k]] + list(aggStats_df[i, j, k]) for
                  i, j, k in topK_idxs]
      topK_os3_df = pd.DataFrame(topK_os3, columns=['OS code1', 'OS code2', 'OS code3',
                                                    'Cooccurrence_Count', 'Mean', 'Median', 'Std'])
      # print("Top %d OS code cooccurrence triplet: \n" % K)
      # topK_os3_df.head(K)
      return topK_os3_df

    elif order == 4:
      sorted_idxs = [k for k, v in sorted(freq_df.items(), key=itemgetter(1), reverse=True)]
      topK_idxs = sorted_idxs[:K]  # K quadruple
      topK_os4 = [[diaglabels[i], diaglabels[j], diaglabels[k], diaglabels[s], freq_df[(i, j, k, s)]] + list(
        aggStats_df[(i, j, k, s)]) for i, j, k, s in topK_idxs]
      topK_os4_df = pd.DataFrame(topK_os4, columns=['OS code1', 'OS code2', 'OS code3', 'OS code 4',
                                                    'Cooccurrence_Count', 'Mean', 'Median', 'Std'])
      return topK_os4_df

    else:
      print("Order %d not supported!" % order)

  elif diagcode == 'CPT':
    pass

  elif diagcode == '':
    pass
