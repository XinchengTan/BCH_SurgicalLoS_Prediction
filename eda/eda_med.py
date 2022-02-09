import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .. import globals


def med_category_vs_los(df: pd.DataFrame, level, level_decile: pd.DataFrame, freq_range=(0, 20), violin=True):
  med_col = globals.DRUG_COLS[level-1]  # column name of the medication type at 'level'
  med_cnt, med_dcl = 'MED%d_COUNT' % level, 'MED%d_DECILE' % level
  med_exp = df[[globals.LOS, med_col]].explode(med_col).dropna(subset=[med_col])
  topK_frequent_med = level_decile.sort_values(by=med_cnt, ascending=False)[freq_range[0]:freq_range[1]]

  med_exp_topK_cnt = med_exp[med_exp[med_col].isin(topK_frequent_med[med_col])]\
    .join(level_decile.set_index(med_col)[[med_cnt, 'MED%d_DECILE' % level]], on=med_col, how='inner')\
    .sort_values(by=med_cnt, ascending=False)

  # Scatter Plot
  fig, ax = plt.subplots(1, 1, figsize=(20, 13))
  if not violin:
    ax.scatter(med_exp_topK_cnt[med_col], med_exp_topK_cnt[globals.LOS], facecolors='none', edgecolors='g')
    ax.plot(med_exp_topK_cnt[med_col], med_exp_topK_cnt[med_dcl], linestyle='None',
            marker="*", markersize=10, markeredgecolor="red", markerfacecolor="red")
  else:
    # White dot: median; thick grey bar: interquantile range (25% - 75%); thin grey bar: rest of data
    sns.violinplot(med_exp_topK_cnt[med_col], med_exp_topK_cnt[globals.LOS], ax=ax)

  ax.set_ylim(-5, 25)
  ax.set_xlabel("Level %d Medication Type" % level, fontsize=15)
  ax.set_ylabel("Length of Stay (days)", fontsize=15)
  ax.set_title("LoS Distribution by Medication Type (frequency range=%s)" % str(freq_range), fontsize=18, y=1.01)
  ax.xaxis.set_tick_params(labelsize=13)
  ax.yaxis.set_tick_params(labelsize=13)
  fig.autofmt_xdate(rotation=45)
  plt.show()

  # Violin Plot


