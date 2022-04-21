from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
from tqdm import tqdm

import globals
from globals import *
from c1_data_preprocessing import gen_y_nnt


# ------------------------------------------- Per-case CCSR count - Outcome -------------------------------------------
def ccsr_cnt_eda(df, outcome=NNT, preprocess_y=True):
  figs, axs = plt.subplots(2, 1, figsize=(16, 18))
  df = df[[outcome, CCSRS]]
  df['ccsr_cnt'] = df[CCSRS].apply(lambda x: len(set(x)))
  if preprocess_y and outcome == NNT:
    df[outcome] = gen_y_nnt(df[outcome])
    axs[1].set_yticks(sorted(df[outcome].unique()))
    axs[1].set_yticklabels(NNT_CLASS_LABELS, fontsize=14)

  cap = 20
  df['capped_cnt0'] = df['ccsr_cnt'].apply(lambda x: x if x < cap else cap)
  counter = Counter(df['capped_cnt0'])
  xs = sorted(counter.keys())
  axs[0].bar(xs, [counter[x] for x in xs], alpha=0.8)
  axs[0].set_title(f'Per-case CCSR Count Histogram', fontsize=18, y=1.01)
  axs[0].set_xlabel(f'CCSR count per case', fontsize=16)
  axs[0].set_ylabel('Number of cases', fontsize=16)
  axs[0].yaxis.set_tick_params(labelsize=14)
  axs[0].xaxis.set_tick_params(labelsize=14)
  axs[0].set_xticks(xs)
  axs[0].set_xticklabels(list(map(int, xs[:-1])) + [r'$\geq$%d' % cap], fontsize=14)
  rects = axs[0].patches
  labels = ["{:.1%}".format(counter[x] / df.shape[0]) for x in xs]
  for rect, label in zip(rects, labels):
    ht, wd = rect.get_height(), rect.get_width()
    axs[0].text(rect.get_x() + wd / 2, ht + 2.5, label,
                ha='center', va='bottom', fontsize=9)

  # cap = 13
  df['capped_cnt1'] = df['ccsr_cnt'].apply(lambda x: x if x < cap else cap)
  sns.violinplot(data=df, x='capped_cnt1', y=outcome)
  axs[1].set_title('LOS Distribution over CCSR Count', fontsize=18, y=1.01)
  axs[1].set_xlabel(f'CCSR count per case', fontsize=16)
  axs[1].set_ylabel('Length of stay', fontsize=16)
  axs[1].yaxis.set_tick_params(labelsize=14)
  axs[1].set_xticklabels(list(map(int, sorted(df['capped_cnt1'].unique())[:-1])) + [r'$\geq$%d' % cap],
                         fontsize=14)


# ------------------------------------------- Top-k CCSR - Outcome -------------------------------------------
def ccsr_eda(dashb_df: pd.DataFrame, outcome=LOS, topK=20, xlim=30):
  total = dashb_df.shape[0]
  df = dashb_df[[LOS, NNT, CCSRS, 'SURG_CASE_KEY']]
  figs, axs = plt.subplots(1, 2, figsize=(27, 11), gridspec_kw={'width_ratios': [5, 2]})
  ax, ax2 = axs[0], axs[1]
  df = df.explode(CCSRS).fillna({CCSRS: 'No CCSRs'})

  groupby = df.groupby(by=CCSRS)
  df['case_cnt'] = groupby['SURG_CASE_KEY'].transform('count')
  df['los_median'] = groupby[outcome].transform('median')
  if xlim:
    df[outcome] = df[outcome].apply(lambda x: x if x < xlim else xlim + random.uniform(0, 0.5))
    ax.set_xlim([0, xlim+0.5])
  pproc_to_los = groupby.agg({outcome: lambda x: list(x),
                              'case_cnt': lambda x: len(x),
                              'los_median': lambda x: max(x),
                              }).reset_index()\
    .sort_values(by='case_cnt', ascending=False)
  pproc_to_los['cumsum_cnt'] = pproc_to_los['case_cnt'].cumsum()
  topK_pproc_dict = pproc_to_los.head(topK).set_index(CCSRS).sort_values(by='los_median').to_dict()
  topK_pproc_to_los = topK_pproc_dict[outcome]

  cmap = plt.cm.get_cmap('tab20')
  colors = cmap(np.linspace(0., 1., topK))

  bplot = ax.boxplot(topK_pproc_to_los.values(), widths=0.7, notch=True, vert=False,
                     patch_artist=True, flierprops={'color': colors})
  ax.set_title(f"LOS Distribution over {topK} Most Common CCSR Diagnoses", fontsize=24, y=1.02)
  ax.set_xlabel("Length of stay", fontsize=20)
  ax.set_ylabel("CCSR diagnosis", fontsize=20)
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

  ax2.set_title("CCSR Diagnosis Histogram", fontsize=24, y=1.02)
  rects = ax2.patches
  labels = ["{:.1%}".format(cnt / total) for cnt in topK_pproc_dict['case_cnt'].values()]
  for rect, label in zip(rects, labels):
    ht, wd = rect.get_height(), rect.get_width()
    ax2.text(wd + 15, rect.get_y() + ht / 2, label,
            ha='left', va='center', fontsize=12)
  ax2.set_xlim([0, 1.1 * max(topK_pproc_dict['case_cnt'].values())])
  ax2.set_ylim([-0.5, topK-0.5])
  plt.tight_layout(w_pad=1)
  plt.savefig('ccsr_top20.png', dpi=500)
  plt.show()

  return df.head(15)

# ------------------------------------------- CCSR-wise LOS median - Outcome -------------------------------------------
# see function in 'clinical_var_risk_score_eda()' in eda_proc.py

# ---------------------------------------- Organ System Code ----------------------------------------
def os_code_colors():
  cmap = plt.cm.get_cmap('tab20')
  colors = cmap(np.linspace(0., 1., len(OS_CODE_LIST) - 1))
  colors = np.array(list(colors) + [colors[6]])
  return colors


# Surgical case histogram VS OS code
def organ_system_eda(dashboard_df, show_pct=False):
  diag_cnts = [dashboard_df[lbl].sum() for lbl in OS_CODE_LIST]

  figs, axs = plt.subplots(1, 2, figsize=(18, 7))
  colors = os_code_colors()
  axs[0].pie(diag_cnts, labels=OS_CODE_LIST, colors=colors, autopct='%.2f%%', startangle=90,
             wedgeprops={"edgecolor": "gray", 'linewidth': 0.3})
  axs[0].set_title("Organ System Diagnosis Code Distribution", y=1.07, fontsize=16)
  axs[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

  dashboard_df['OS_code_count'] = dashboard_df[OS_CODE_LIST].sum(axis=1)
  print("Number of patients with more than 1 diagnosed conditions: ",
        dashboard_df[dashboard_df['OS_code_count'] > 1].shape[0])

  os_counter = Counter(dashboard_df['OS_code_count'])
  xs = sorted(os_counter.keys())
  axs[1].bar(xs, [os_counter[x] for x in xs],
             color="purple", alpha=0.6, edgecolor="black", linewidth=0.5)
  axs[1].set_title("Per-case Body System Diagnoses Count Distribution", fontsize=16, y=1.05)
  axs[1].set_xlabel("Number of body system diagnoses per case", fontsize=14)
  axs[1].set_ylabel("Number of cases", fontsize=14)
  axs[1].set_xticks(xs)
  rects = axs[1].patches
  if show_pct:
    labels = ["{:.1%}".format(os_counter[x] / dashboard_df.shape[0]) for x in xs]
  else:
    labels = [str(os_counter[x]) for x in xs]  # "{:.1%}".format(outcome_cnter[i] / total_cnt) for i in xs
  for rect, label in zip(rects, labels):
    ht, wd = rect.get_height(), rect.get_width()
    axs[1].text(rect.get_x() + wd / 2, ht + 2.5, label,
                ha='center', va='bottom', fontsize=9)

  # for i in bins[:-1]:
  #   plt.text(arr[1][i],arr[0][i],str(int(arr[0][i])))
  plt.show()


# Organ system code & LOS
def organ_system_los_boxplot(dashboard_df, exclude_outliers=0, xlim=None):
  # Box plot of LoS distribution over all 21 OS codes
  #dashboard_df = utils.drop_outliers(dashboard_df, exclude_outliers)

  diag2los = defaultdict(list)
  for diag in OS_CODE_LIST:
    diag2los[diag] = dashboard_df[dashboard_df[diag] == 1].LENGTH_OF_STAY.tolist()
  dlabels = [k for k, _ in sorted(diag2los.items(), key=lambda kv: np.median(kv[1]))]

  fig, ax = plt.subplots(figsize=(25, 17))
  colors = os_code_colors()
  colors = [colors[OS_CODE_LIST.index(diag)] for diag in dlabels]

  bplot = ax.boxplot([diag2los[d] for d in dlabels], widths=0.7, notch=True, vert=False,
                     patch_artist=True, flierprops={'color': colors})
  ax.set_title("LOS Distribution over Body System Diagnosis", fontsize=22, y=1.01)
  ax.set_xlabel("LOS (day)", fontsize=18)
  ax.set_yticklabels(dlabels, fontsize=14)
  ax.set_ylabel("Body system diagnosis", fontsize=18)
  ax.xaxis.set_tick_params(labelsize=14)
  if xlim:
    ax.set_xlim([0, xlim])

  for patch, color in zip(bplot["fliers"], colors):
    patch.set_markeredgecolor(color)
    patch.set_markeredgewidth(2)

  for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)




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


# Generates a dataframe mapping each medical code to its count and frequency
def medical_code_hist(Xdf, code):
  N = Xdf.shape[0]
  if code not in ['PRIMARY_PROC', 'CPT', 'CCSR']:
    raise NotImplementedError

  cols = list(filter(lambda x: x.startswith(code + '_'), Xdf.columns.to_list()))
  count = Xdf[cols].sum(axis=0)
  hist_df = pd.DataFrame(count, index=cols, columns=['Count'])
  hist_df['Count Ratio'] = hist_df['Count'] / N

  return


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
  #ret_df.index.set_names(names=['Primary Procedure', 'CCSR'], inplace=True)

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