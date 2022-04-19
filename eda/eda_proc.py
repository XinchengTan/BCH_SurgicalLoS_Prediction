
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
# import hvplot
# import panel as pn

from collections import Counter, defaultdict
from globals import *
from c1_data_preprocessing import gen_y_nnt


# ------------------------------------------ CPT Count - Outcome ------------------------------------------
def cpt_count_eda(dashb_df, outcome=NNT, preprocess_y=True):
  figs, axs = plt.subplots(2, 1, figsize=(12, 16))
  df = dashb_df[[outcome, CPTS]]
  df['cpt_cnt'] = df[CPTS].apply(lambda x: len(set(x)))
  if preprocess_y and outcome == NNT:
    df[outcome] = gen_y_nnt(df[outcome])
    axs[1].set_yticks(sorted(df[outcome].unique()))
    axs[1].set_yticklabels(NNT_CLASS_LABELS, fontsize=14)

  cap = 6
  df['capped_cnt0'] = df['cpt_cnt'].apply(lambda x: x if x < cap else cap)
  counter = Counter(df['capped_cnt0'])
  xs = sorted(counter.keys())
  axs[0].bar(xs, [counter[x] for x in xs], alpha=0.8)
  axs[0].set_title(f'Per-case CPT Count Histogram', fontsize=18, y=1.01)
  axs[0].set_xlabel(f'CPT count per case', fontsize=16)
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
  df['capped_cnt1'] = df['cpt_cnt'].apply(lambda x: x if x < cap else cap)
  sns.violinplot(data=df, x='capped_cnt1', y=outcome)
  axs[1].set_title('LOS Distribution over CPT Count', fontsize=18, y=1.01)
  axs[1].set_xlabel(f'CPT count per case', fontsize=16)
  axs[1].set_ylabel('Length of stay', fontsize=16)
  axs[1].yaxis.set_tick_params(labelsize=14)
  axs[1].set_xticklabels(list(map(int, sorted(df['capped_cnt1'].unique())[:-1])) + [r'$\geq$%d' % cap],
                         fontsize=14)


# ----------------------------------------- Top-k Primary Procedure - Outcome -----------------------------------------
def pproc_eda(dashb_df: pd.DataFrame, outcome=LOS, topK=20, xlim=30):
  total = dashb_df.shape[0]
  df = dashb_df[[LOS, NNT, PRIMARY_PROC, 'SURG_CASE_KEY']]
  figs, axs = plt.subplots(1, 2, figsize=(27, 11), gridspec_kw={'width_ratios': [5, 2]})
  ax, ax2 = axs[0], axs[1]
  if xlim:
    df[outcome] = df[outcome].apply(lambda x: x if x < xlim else xlim+random.uniform(0, 0.5))
    ax.set_xlim([0, xlim+0.5])
  groupby = df.groupby(by=PRIMARY_PROC)
  df['case_cnt'] = groupby['SURG_CASE_KEY'].transform('count')
  df['los_median'] = groupby[outcome].transform('median')
  pproc_to_los = groupby.agg({outcome: lambda x: list(x),
                              'case_cnt': lambda x: len(x),
                              'los_median': lambda x: max(x),
                              }).reset_index()\
    .sort_values(by='case_cnt', ascending=False)
  pproc_to_los['cumsum_cnt'] = pproc_to_los['case_cnt'].cumsum()
  topK_pproc_dict = pproc_to_los.head(topK).set_index(PRIMARY_PROC).sort_values(by='los_median').to_dict()
  topK_pproc_to_los = topK_pproc_dict[outcome]

  cmap = plt.cm.get_cmap('tab20')
  colors = cmap(np.linspace(0., 1., topK))

  bplot = ax.boxplot(topK_pproc_to_los.values(), widths=0.7, notch=True, vert=False,
                     patch_artist=True, flierprops={'color': colors})
  ax.set_title(f"LOS Distribution over {topK} Most Common Primary Procedures", fontsize=24, y=1.02)
  ax.set_xlabel("Length of stay", fontsize=20)
  ax.set_ylabel("Primary procedure", fontsize=20)
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

  ax2.set_title("Primary Procedure Histogram", fontsize=24, y=1.02)
  rects = ax2.patches
  labels = ["{:.1%}".format(cnt / total) for cnt in topK_pproc_dict['case_cnt'].values()]
  for rect, label in zip(rects, labels):
    ht, wd = rect.get_height(), rect.get_width()
    ax2.text(wd + 15, rect.get_y() + ht / 2, label,
            ha='left', va='center', fontsize=11)
  ax2.set_xlim([0, 1.1 * max(topK_pproc_dict['case_cnt'].values())])
  ax2.set_ylim([-0.5, topK-0.5])
  plt.tight_layout(w_pad=1)
  plt.savefig('pproc_top20.png', dpi=500)
  plt.show()


# --------------------------------------- Primary Procedure LOS Median - Outcome ---------------------------------------
def pproc_los_median_eda(dashb_df, outcome=NNT, preprocess_y=True, min_cnt=5, xlim=15, save=False):
  total = dashb_df.shape[0]
  df = dashb_df[[LOS, NNT, PRIMARY_PROC, 'SURG_CASE_KEY']]
  figs, axs = plt.subplots(1, 1, figsize=(12, 9))  # gridspec_kw={'width_ratios': [2, 1]}
  #ax, ax2 = axs[0], axs[1]
  ax = axs
  if preprocess_y:
    df[outcome] = gen_y_nnt(df[outcome])
    ax.set_yticks(sorted(df[outcome].unique()))
    ax.set_yticklabels(NNT_CLASS_LABELS, fontsize=13)

  groupby = df.groupby(by=PRIMARY_PROC)
  pproc_to_los_stat = groupby[LOS].median().reset_index(name='los_median').join(
    groupby['SURG_CASE_KEY'].count().reset_index(name='case_cnt').set_index(PRIMARY_PROC),
    on=PRIMARY_PROC,
    how='inner'
  )
  pproc_to_los_stat = pproc_to_los_stat[pproc_to_los_stat['case_cnt'] >= min_cnt]
  print('Remaining cases: ', pproc_to_los_stat['case_cnt'].sum())

  df_filtered = df.join(pproc_to_los_stat.set_index(PRIMARY_PROC), on=PRIMARY_PROC, how='inner')
  outlier_df = df_filtered[df_filtered['los_median'] >= xlim]
  if xlim:
    df_filtered[outcome] = df_filtered[outcome].apply(lambda x: x if x < xlim else xlim+random.uniform(0, 0.5))
    ax.set_xlim([-0.8, xlim+0.8])
  #sns.boxplot(data=df_filtered, x='los_median', y=NNT, orient='h', ax=ax)
  sns.violinplot(data=df_filtered, x='los_median', y=NNT, orient='h', ax=ax, scale='width')
  #ax.scatter(outliers_df['los_median'], outliers_df[NNT], facecolors='none', edgecolors='r')
  ax.set_title('Primary Procedure LOS Median vs LOS Outcome', fontsize=18, y=1.01)
  ax.set_xlabel('LOS median per primary procedure', fontsize=16)
  ax.set_ylabel('LOS (outcome class)', fontsize=16)
  ax.set_yticks(sorted(df[outcome].unique()))
  ax.set_yticklabels(NNT_CLASS_LABELS, fontsize=14)
  ax.xaxis.set_tick_params(labelsize=14)
  ax.invert_yaxis()
  plt.setp(ax.collections, alpha=.9)
  if save:
    plt.savefig('pproc_los_med_outcome.png', dpi=400)
  return outlier_df
  #return df_filtered.head(10)
  #df['case_cnt'] = groupby['SURG_CASE_KEY'].transform('count')

  # outcome_cnter = Counter(y)
  # ys = sorted(outcome_cnter.keys())
  # ax.barh(ys, [outcome_cnter[i] for i in ys], align='center')
  # ax.set_xlabel("Number of surgical cases", fontsize=16)
  # ax.invert_yaxis()
  # ax.set_yticks(ys)
  # ax.set_yticklabels(NNT_CLASS_LABELS, fontsize=13)
  # ax.set_title("LoS Histogram (%s)" % dataType)
  # rects = ax.patches
  # total_cnt = len(y)
  # labels = ["{:.1%}".format(outcome_cnter[i] / total_cnt) for i in ys]
  # for rect, label in zip(rects, labels):
  #   ht, wd = rect.get_height(), rect.get_width()
  #   ax.text(wd + 2.5, rect.get_y() + ht / 2, label,
  #           ha='left', va='center', fontsize=15)
  # ax2.scatter([1,1], [1,1])

  plt.show()

