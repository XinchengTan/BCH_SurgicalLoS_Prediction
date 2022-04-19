import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
from collections import Counter

import globals
from globals import *
from c1_data_preprocessing import gen_y_nnt


# ------------------------------------------ Medication Type Count - Outcome ------------------------------------------
def med_count_eda(dashb_df, level, outcome=NNT, preprocess_y=True, max_cnt=5):
  figs, axs = plt.subplots(2, 1, figsize=(12, 16))
  MED_col = DRUG_COLS[level-1]
  if level == 4:
    level = '3 or below'
  df = dashb_df[[outcome, MED_col]]
  df['med_cnt'] = df[MED_col].apply(lambda x: len(set(x)))
  if preprocess_y and outcome == NNT:
    df[outcome] = gen_y_nnt(df[outcome])
    axs[1].set_yticks(sorted(df[outcome].unique()))
    axs[1].set_yticklabels(NNT_CLASS_LABELS, fontsize=14)

  cap = max_cnt
  df['capped_cnt0'] = df['med_cnt'].apply(lambda x: x if x < cap else cap)
  counter = Counter(df['capped_cnt0'])
  xs = sorted(counter.keys())
  axs[0].bar(xs, [counter[x] for x in xs], alpha=0.8)
  axs[0].set_title(f'Medication (Level-{level}) Count Histogram', fontsize=18, y=1.01)
  axs[0].set_xlabel(f'Level-{level} medication count per case', fontsize=16)
  axs[0].set_ylabel('Number of cases', fontsize=16)
  axs[0].yaxis.set_tick_params(labelsize=14)
  axs[0].xaxis.set_tick_params(labelsize=14)
  axs[0].set_xticks(xs)
  axs[0].set_xticklabels(list(map(int, xs[:-1])) + [r'$\geq$%d' % cap], fontsize=14)
  rects = axs[0].patches
  labels = ["{:.1%}".format(counter[x] / df.shape[0]) for x in xs]
  for rect, label in zip(rects, labels):
    ht, wd = rect.get_height(), rect.get_width()
    axs[0].text(rect.get_x() + wd / 2, ht + 35, label,
                ha='center', va='bottom', fontsize=12)

  # cap = 13
  df['capped_cnt1'] = df['med_cnt'].apply(lambda x: x if x < cap else cap)
  sns.violinplot(data=df, x='capped_cnt1', y=outcome, scale='width', width=0.75)
  axs[1].set_title(f'LOS Distribution over Medication (Level-{level}) Count', fontsize=18, y=1.01)
  axs[1].set_xlabel(f'Level-{level} medication category count per case', fontsize=16)
  axs[1].set_ylabel('Length of stay', fontsize=16)
  axs[1].yaxis.set_tick_params(labelsize=14)
  axs[1].set_xticklabels(list(map(int, sorted(df['capped_cnt1'].unique())[:-1])) + [r'$\geq$%d' % cap],
                         fontsize=14)
  plt.tight_layout(h_pad=6)
  plt.savefig('med_cnt.png', dpi=300)

# ----------------------------------------- Top-k Medication - Outcome -----------------------------------------
def med_eda(dashb_df: pd.DataFrame, med_level, outcome=LOS, topK=20, xlim=30):
  MED_col = DRUG_COLS[med_level - 1]
  if med_level == 4:
    med_level = '3 or below'
  total = dashb_df.shape[0]
  df = dashb_df[[LOS, NNT, MED_col, 'SURG_CASE_KEY']]

  figs, axs = plt.subplots(1, 2, figsize=(27, 11), gridspec_kw={'width_ratios': [5, 2]})
  ax, ax2 = axs[0], axs[1]

  df = df.explode(MED_col).fillna({MED_col: 'no medication'})
  groupby = df.groupby(by=MED_col)
  df['case_cnt'] = groupby['SURG_CASE_KEY'].transform('count')
  df['los_median'] = groupby[outcome].transform('median')
  if xlim:
    df[outcome] = df[outcome].apply(lambda x: x if x < xlim else xlim+random.uniform(0, 0.5))
    ax.set_xlim([0, xlim+0.5])
  pproc_to_los = groupby.agg({outcome: lambda x: list(x),
                              'case_cnt': lambda x: len(x),
                              'los_median': lambda x: max(x),
                              }).reset_index()\
    .sort_values(by='case_cnt', ascending=False)
  pproc_to_los['cumsum_cnt'] = pproc_to_los['case_cnt'].cumsum()
  topK_pproc_dict = pproc_to_los.head(topK).set_index(MED_col).sort_values(by='los_median').to_dict()
  topK_pproc_to_los = topK_pproc_dict[outcome]

  cmap = plt.cm.get_cmap('tab20')
  colors = cmap(np.linspace(0., 1., topK))

  bplot = ax.boxplot(topK_pproc_to_los.values(), widths=0.7, notch=True, vert=False,
                     patch_artist=True, flierprops={'color': colors})
  ax.set_title(f"LOS Distribution over {topK} Most Common Level-{med_level} Medication", fontsize=24, y=1.02)
  ax.set_xlabel("Length of stay", fontsize=20)
  ax.set_ylabel(f"Level-{med_level} medication type", fontsize=20)
  ax.set_xticks(np.arange(0, xlim+1))
  ax.set_xticklabels(list(map(str, np.arange(0, xlim))) + [r'$\geq$%d'%xlim], fontsize=15)
  ax.set_yticklabels(topK_pproc_to_los.keys(), fontsize=15)

  for patch, color in zip(bplot["fliers"], colors):
    patch.set_markeredgecolor(color)
    patch.set_markeredgewidth(2)

  for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

  # fig, ax2 = plt.subplots(figsize=(6, 9))
  ax2.barh(range(topK), topK_pproc_dict['case_cnt'].values(), align='center', alpha=0.85, height=0.7)
  ax2.set_xlabel("Number of cases", fontsize=19)
  #ax.invert_yaxis()
  ax2.set_yticks(range(topK))
  ax2.set_yticklabels([''] * topK, fontsize=13)
  ax2.xaxis.set_tick_params(labelsize=13)

  ax2.set_title(f"Level-{med_level} Medication Histogram", fontsize=24, y=1.02)
  rects = ax2.patches
  labels = ["{:.1%}".format(cnt / total) for cnt in topK_pproc_dict['case_cnt'].values()]
  for rect, label in zip(rects, labels):
    ht, wd = rect.get_height(), rect.get_width()
    ax2.text(wd + 15, rect.get_y() + ht / 2, label,
            ha='left', va='center', fontsize=11)
  ax2.set_xlim([0, 1.1 * max(topK_pproc_dict['case_cnt'].values())])
  ax2.set_ylim([-0.5, topK-0.5])
  plt.tight_layout(w_pad=1)
  plt.savefig('med_top20.png', dpi=500)
  plt.show()



def med_category_vs_los(data_df: pd.DataFrame, level, level_decile: pd.DataFrame, freq_range=(0, 20),
                        outcome=LOS, preprocess_y=False, violin=True):
  fig, ax = plt.subplots(1, 1, figsize=(20, 13))
  med_col = DRUG_COLS[level-1]  # column name of the medication type at 'level'
  med_cnt, med_dcl, med_median = f'MED{level}_COUNT', f'MED{level}_DECILE', f'MED{level}_MEDIAN'
  df = data_df[[outcome, med_col]]
  if outcome == NNT and preprocess_y:
    df[outcome] = gen_y_nnt(df[outcome])
    ax.set_yticks(sorted(df[outcome].unique()))
    ax.set_yticklabels(NNT_CLASS_LABELS, fontsize=14)
  else:
    ax.set_ylim(-3, 10)
  #print(df[outcome].unique())
  med_exp = df.explode(med_col).dropna(subset=[med_col])
  topK_frequent_med = level_decile.sort_values(by=med_cnt, ascending=False)[freq_range[0]:freq_range[1]]

  med_exp_topK_cnt = med_exp[med_exp[med_col].isin(topK_frequent_med[med_col])]\
    .join(level_decile.set_index(med_col)[[med_cnt, med_median, 'MED%d_DECILE' % level]], on=med_col, how='inner')\
    .sort_values(by=med_median)

  if violin:
    # Violin Plot
    # White dot: median; thick grey bar: interquartile range (25% - 75%); thin grey bar: rest of data
    sns.violinplot(med_exp_topK_cnt[med_col], med_exp_topK_cnt[outcome], ax=ax)
    # ax.plot(med_exp_topK_cnt[med_col], med_exp_topK_cnt[med_dcl], linestyle='None',
    #         marker="*", markersize=10, markeredgecolor="tomato", markerfacecolor="tomato")
  else:
    # Scatter Plot
    ax.scatter(med_exp_topK_cnt[med_col], med_exp_topK_cnt[outcome], facecolors='none', edgecolors='g')
    ax.plot(med_exp_topK_cnt[med_col], med_exp_topK_cnt[med_dcl], linestyle='None',
            marker="*", markersize=10, markeredgecolor="red", markerfacecolor="red")

  ax.set_xlabel("Level-%d medication type" % level, fontsize=16)
  ax.set_ylabel("Length of stay", fontsize=16)
  ax.set_title(f"LOS Distribution by Level-{level} Medication Type (frequency range={freq_range})", fontsize=20, y=1.01)
  ax.set_xticks(np.arange(freq_range[0], freq_range[1]))
  xtick_labels = med_exp_topK_cnt[med_col].unique()
  xtick_suffix = [level_decile.set_index(med_col).loc[med, med_cnt] for med in xtick_labels]
  ax.set_xticklabels([f'{xtick_labels[i]}\n($n$={xtick_suffix[i]})'
                      for i in range(freq_range[0], freq_range[1])], fontsize=13)
  ax.yaxis.set_tick_params(labelsize=14)
  fig.autofmt_xdate(rotation=45)
  plt.show()



