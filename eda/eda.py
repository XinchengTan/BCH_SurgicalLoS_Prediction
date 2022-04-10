"""
EDA on demographics features
"""
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from globals import *
from c1_data_preprocessing import preprocess_y
import utils_plot
import utils


# -------------------------------------------- LOS --------------------------------------------
def los_histogram(y, dataType='Training', ax=None):
  if ax is None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
  outcome_cnter = Counter(y)
  ax.barh(range(MAX_NNT + 2), [outcome_cnter[i] for i in range(MAX_NNT + 2)], align='center')
  ax.set_xlabel("Number of surgical cases")
  ax.invert_yaxis()
  ax.set_yticks(range(MAX_NNT + 2))
  ax.set_yticklabels(NNT_CLASS_LABELS, fontsize=13)
  ax.set_title("LoS Histogram (%s)" % dataType)
  rects = ax.patches
  total_cnt = len(y)
  labels = ["{:.1%}".format(outcome_cnter[i] / total_cnt) for i in range(MAX_NNT + 2)]
  for rect, label in zip(rects, labels):
    ht, wd = rect.get_height(), rect.get_width()
    ax.text(wd + 2.5, rect.get_y() + ht / 2, label,
                ha='left', va='center', fontsize=15)


def los_histogram_vert(y, ax=None, outcome=LOS, clip_y=False):
  if ax is None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
  y = np.array(y)
  if clip_y:
    y[y > MAX_NNT] = MAX_NNT + 1

  outcome_cnter = Counter(y)
  ax.bar(range(MAX_NNT + 2), [outcome_cnter[i] for i in range(MAX_NNT + 2)], align='center', alpha=0.85)
  ax.set_xlabel(outcome.replace('_', ' '), fontsize=14)
  ax.set_ylabel("Number of surgical cases", fontsize=14)
  ax.set_xticks(range(MAX_NNT + 2))
  ax.set_xticklabels(NNT_CLASS_LABELS, fontsize=13)
  ax.set_title(f"{outcome.replace('_', ' ')} Histogram", fontsize=16, y=1.01)
  rects = ax.patches
  total_cnt = len(y)
  labels = ["{:.1%}".format(outcome_cnter[i] / total_cnt) for i in range(MAX_NNT + 2)]
  for rect, label in zip(rects, labels):
    ht, wd = rect.get_height(), rect.get_width()
    ax.text(rect.get_x() + wd / 2, ht + 2.5, label,
            ha='center', va='bottom', fontsize=13)


def los_histogram_by_year(df, outcome=NNT, clip_y=False, show_pct=False, os_df=None, os_aside=False, ax=None):
  if os_df is not None:
    df = pd.concat([df, os_df], axis=0)

  y_df = df[['SURG_CASE_KEY', outcome, ADMIT_DTM]]
  if clip_y:
    y_df.loc[:, outcome] = y_df[outcome].apply(lambda x: MAX_NNT + 1 if x > MAX_NNT else x)
  y_df.loc[:, ADMIT_YEAR] = y_df[ADMIT_DTM].dt.year
  if os_df is not None and os_aside:
    y_df.loc[y_df['SURG_CASE_KEY'].isin(os_df['SURG_CASE_KEY']), ADMIT_YEAR] = 3000  # placeholder

  yr_nnt_cnt_df = y_df.groupby(by=[ADMIT_YEAR, outcome]).size().to_frame(name='count').reset_index(drop=False)
  yr_nnt_cnt_df.loc[:, 'annual_total'] = yr_nnt_cnt_df.groupby(ADMIT_YEAR)['count'].transform('sum')
  yr_nnt_cnt_df.loc[:, 'count_pct'] = 100 * yr_nnt_cnt_df['count'] / yr_nnt_cnt_df['annual_total']
  display_col = 'count_pct' if show_pct else 'count'
  ylabel = 'Annual Surgical Case Count Percentage (%)' if show_pct else 'Number of Surgical Cases'

  if ax is None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
  year_set = sorted(yr_nnt_cnt_df[ADMIT_YEAR].unique())
  year_label = ['Out-of-sample' if yr == 3000 else yr for yr in year_set]
  width = 0.8 / len(year_set)
  for yr_idx in range(len(year_set)):
    cur_nnt_cnt = yr_nnt_cnt_df.loc[yr_nnt_cnt_df[ADMIT_YEAR] == year_set[yr_idx]]
    xs = cur_nnt_cnt[outcome].to_numpy() + (2 * yr_idx - 3) * width / 2
    ax.bar(xs, cur_nnt_cnt[display_col], width=width, label=year_label[yr_idx], alpha=0.78)
  ax.set_ylabel(ylabel, fontsize=15, x=-0.1)
  ax.set_xlabel(outcome.replace('_', ' '), fontsize=15)
  ax.yaxis.set_tick_params(labelsize=13)
  ax.set_xticks(np.arange(MAX_NNT + 2))
  ax.set_xticklabels(NNT_CLASS_LABELS, fontsize=13)
  ax.set_title(f"{outcome.replace('_', ' ')} Histogram by Year", fontsize=16, y=1.01)
  ax.legend(prop={'size': 13})


def los_histogram_by_care_class(df, outcome=NNT, clip_y=False, pct=False, ax=None):
  y_df = df[[outcome, 'CARE_CLASS']]
  if clip_y:
    y_df.loc[:, outcome] = y_df[outcome].apply(lambda x: MAX_NNT + 1 if x > MAX_NNT else x)
  if ax is None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
  if pct == 'global':
    ylabel = 'Global Frequency (%)'
    normalizer = df.shape[0]
  elif pct == 'group':
    ylabel = 'Group-wise Frequency (%)'
    normalizer = np.nan
  else:
    ylabel = 'Count'
    normalizer = 1

  groupby = y_df.groupby(by='CARE_CLASS')
  width = 0.8 / len(groupby)
  idx = 1
  care_cls_to_counter_graph = {}
  for care_cls, c_df in groupby:
    print(care_cls, c_df.shape[0])
    xs = np.sort(c_df[outcome].unique()) + (2 * idx - 3) * width / 2
    outcome_counter = defaultdict(int)
    outcome_counter.update(Counter(c_df[outcome].to_numpy()))
    normalizer = c_df.shape[0] if pct == 'group' else normalizer
    graph = ax.bar(xs, [outcome_counter[c] / normalizer for c in sorted(c_df[outcome].unique())],
                   width=width, label=care_cls, alpha=0.8)
    care_cls_to_counter_graph[care_cls] = (outcome_counter, graph)
    idx += 1
  ax.set_ylabel(ylabel, fontsize=15, x=-0.1)
  ax.set_xlabel(outcome.replace('_', ' '), fontsize=15)
  ax.yaxis.set_tick_params(labelsize=13)
  ax.set_xticks(np.arange(MAX_NNT + 2))
  ax.set_xticklabels(NNT_CLASS_LABELS, fontsize=13)
  ax.set_title(f'{NNT.replace("_", " ")} Distribution Over Care Class', fontsize=16, y=1.01)
  ax.legend(prop={'size': 13})

  if pct:
    for care_cls, (counter, graph) in care_cls_to_counter_graph.items():
      i = 0
      keys = sorted(counter.keys())
      total = sum(counter.values()) if pct == 'group' else df.shape[0]
      for p in graph:
        cnt = counter[keys[i]]
        if cnt > 0:
          width = p.get_width()
          height = p.get_height()
          x, y = p.get_xy()
          plt.text(x + width / 2,
                   y + height + 0.006,
                   "{:.2%}".format(cnt / total),
                   ha='center', fontsize=9)
        i += 1


# ---------------------------------------- Gender & LOS ----------------------------------------
def gender_eda(dashboard_df, outcome=LOS, clip_y=False):
  labels = ["Male", "Female"]
  df = dashboard_df[[GENDER, outcome]]
  gender_cnts = [df.loc[df.SEX_CODE == "M"].shape[0],
                 df.loc[df.SEX_CODE == "F"].shape[0]]

  figs, axs = plt.subplots(1, 2, figsize=(18, 6))
  axs[0].pie(gender_cnts, labels=labels, autopct='%.2f%%', startangle=90, textprops={'fontsize': 17})
  axs[0].set_title("Case Count(%) by Gender", fontsize=18)
  axs[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

  if outcome == NNT and clip_y:
    df.loc[:, outcome] = df[outcome].apply(lambda x: MAX_NNT + 1 if x > MAX_NNT else x)
    counter_male = Counter(df.loc[df.SEX_CODE == "M", outcome])
    counter_female = Counter(df.loc[df.SEX_CODE == "F", outcome])
    axs[1].bar(np.arange(MAX_NNT + 2)-0.2,
               [counter_male[i] for i in range(MAX_NNT + 2)],
               width=0.4, alpha=0.78, label='Male')
    axs[1].bar(np.arange(MAX_NNT + 2)+0.2,
               [counter_female[i] for i in range(MAX_NNT + 2)],
               width=0.4,alpha=0.78, label='Female')
    axs[1].set_xticks(np.arange(MAX_NNT + 2))
    axs[1].set_xticklabels(NNT_CLASS_LABELS, fontsize=13)
  else:
    bins = np.linspace(0, 21, 40)
    axs[1].set_xlim([bins[0], bins[-1]])
    axs[1].hist([df.loc[df.SEX_CODE == "M", outcome],
                 df.loc[df.SEX_CODE == "F", outcome]],
                bins, alpha=0.7, edgecolor='grey', linewidth=0.5, label=["Male", "Female"])
  axs[1].set_title(f"{outcome.replace('_', ' ')} Distribution by Gender", fontsize=18, y=1.01)
  axs[1].set_xlabel("Length of Stay (day)", fontsize=15)
  axs[1].set_ylabel("Number of patients", fontsize=15)

  plt.legend()
  plt.show()


# ---------------------------------------- Age & LOS ----------------------------------------
def age_los_eda(dashboard_df, age_bins=None):
  max_age, min_age = np.ceil(max(dashboard_df.AGE_AT_PROC_YRS) + DELTA), np.floor(min(dashboard_df.AGE_AT_PROC_YRS))
  age_bins = np.linspace(int(min_age), int(max_age), int(np.ceil(max_age - min_age))) if age_bins is None else age_bins

  figs, axs = plt.subplots(3, 1, figsize=(9, 16))
  axs[0].hist(dashboard_df.AGE_AT_PROC_YRS, bins=age_bins, alpha=0.7, edgecolor="black", linewidth=0.5)
  axs[0].set_title("Age distribution")
  axs[0].set_xlabel("Age (yr)")
  axs[0].set_ylabel("Number of patients")

  # Average and median LoS of each age group
  avglos_age = [0] * (len(age_bins) - 1)
  medlos_age = [0] * (len(age_bins) - 1)
  for i in range(len(age_bins) - 1):
    l, h = age_bins[i:i + 2]
    los = dashboard_df[(l <= dashboard_df.AGE_AT_PROC_YRS) & (dashboard_df.AGE_AT_PROC_YRS < h)].LENGTH_OF_STAY
    avglos_age[i] = np.mean(los)
    medlos_age[i] = np.median(los)

  axs[1].bar(age_bins[:-1] + 0.5, avglos_age, alpha=0.5, color="red", edgecolor="black", linewidth=0.5)
  axs[1].set_title("Average LoS in each Age Group")
  axs[1].set_xlabel("Age (yr)")
  axs[1].set_ylabel("LoS")

  axs[2].bar(age_bins[:-1] + 0.5, medlos_age, alpha=0.7, color="orange", edgecolor="black", linewidth=0.5)
  axs[2].set_title("Median LoS in each Age Group")
  axs[2].set_xlabel("Age (yr)")
  axs[2].set_ylabel("LoS")
  figs.tight_layout(pad=3.0)

  print("Correlation between age and LoS: ",
        np.corrcoef(dashboard_df.AGE_AT_PROC_YRS, dashboard_df.LENGTH_OF_STAY)[0, 1])


def age_los_boxplot(df, age_bins=None, ylim=None, outcome=LOS, clip_y=False, violin=False):
  assert outcome in {LOS, NNT}
  if clip_y:
    df = df[[outcome, AGE]]
    df[outcome] = df[outcome].apply(lambda x: x if x <= MAX_NNT else MAX_NNT + 1)

  if not np.isinf(age_bins[-1]):
    age_bins.append(float('-inf'))

  age2los = defaultdict()
  for i in range(len(age_bins) - 1):
    l, h = age_bins[i:i + 2]
    age2los[i] = df[(l <= df[AGE]) & (df[AGE] < h)][outcome].tolist()

  fig, ax = plt.subplots(figsize=(15, 10))
  if violin:
    sns.violinplot(data=[age2los[k] for k in sorted(age2los.keys())], ax=ax)
  else:
    ax.boxplot([age2los[i] for i in sorted(age2los.keys())], widths=0.7, notch=True, patch_artist=True)
  ax.set_title(f"{outcome} Distribution by Age Group", fontsize=19, y=1.01)
  ax.set_xlabel("Age (year)", fontsize=16)
  ax.set_ylabel("LoS (day)", fontsize=16) if outcome == LOS else ax.set_ylabel("Number of Nights", fontsize=16)
  if np.isinf(age_bins[-1]):
    labels = [f'{age_bins[i]}-{age_bins[i+1]}\n$n$={len(age2los[i])}' for i in range(len(age_bins)-2)]
    labels.append(f'{age_bins[-2]}+\n$n$={len(age2los[max(age2los.keys())])}')
  else:
    labels = [f'{age_bins[i]}-{age_bins[i + 1]}' for i in range(len(age_bins) - 1)]
  ax.set_xticklabels(labels, fontsize=13)
  ax.yaxis.set_tick_params(labelsize=14)
  if ylim:
    ax.set_ylim([-0.6, ylim])

  plt.show()


# ---------------------------------------- Weight z-score & LOS ----------------------------------------
def weightz_los_eda(dashboard_df):
  max_wt, min_wt = int(np.ceil(max(dashboard_df.WEIGHT_ZSCORE) + DELTA)), int(np.floor(min(dashboard_df.WEIGHT_ZSCORE)))
  wt_bins = np.linspace(min_wt, max_wt, int(np.ceil(max_wt - min_wt)) * 2)

  figs, axs = plt.subplots(3, 1, figsize=(9, 16))
  axs[0].hist(dashboard_df.WEIGHT_ZSCORE, bins=wt_bins, alpha=0.7, edgecolor="black", linewidth=0.5)
  axs[0].set_title("Weight z-score distribution")
  axs[0].set_xlabel("Weight z-score")
  axs[0].set_ylabel("Number of patients")

  # Average and median LoS of each weight z-score group
  avglos_wt = [0] * (len(wt_bins) - 1)
  medlos_wt = [0] * (len(wt_bins) - 1)
  for i in range(len(wt_bins) - 1):
    l, h = wt_bins[i:i + 2]
    los = dashboard_df[(l <= dashboard_df.WEIGHT_ZSCORE) & (dashboard_df.WEIGHT_ZSCORE < h)].LENGTH_OF_STAY
    if len(los) > 0:
      avglos_wt[i] = np.mean(los)
      medlos_wt[i] = np.median(los)

  axs[1].bar(wt_bins[:-1], avglos_wt, alpha=0.5, width=0.49, color="red", edgecolor="black", linewidth=0.5)
  axs[1].set_title("Average LoS in each Weight z-score Group")
  axs[1].set_xlabel("Weight z-score")
  axs[1].set_ylabel("LoS")

  axs[2].bar(wt_bins[:-1], medlos_wt, alpha=0.7, width=0.49, color="orange", edgecolor="black", linewidth=0.5)
  axs[2].set_title("Median LoS in each Weight z-score Group")
  axs[2].set_xlabel("Weight z-score")
  axs[2].set_ylabel("LoS")
  figs.tight_layout(pad=3.0)

  print("Correlation between age and LoS: ", np.corrcoef(dashboard_df.WEIGHT_ZSCORE, dashboard_df.LENGTH_OF_STAY)[0, 1])


def weightz_los_boxplot(df, weightz_bins, bin_width=1, ylim=None, outcome=LOS, clip_y=False, violin=False):
  assert outcome in {LOS, NNT}
  if clip_y:
    df = df[[outcome, WEIGHT_ZS]]
    df[outcome] = df[outcome].apply(lambda x: x if x <= MAX_NNT else MAX_NNT + 1)

  wz2los = defaultdict()

  if not np.isinf(weightz_bins[0]):
    weightz_bins.insert(0,  float('-inf'))
  if not np.isinf(weightz_bins[-1]):
    weightz_bins.insert(-1, float('inf'))

  for i in range(len(weightz_bins)-1):
    l, h = weightz_bins[i], weightz_bins[i+1]
    wz2los[i] = df.loc[(l <= df[WEIGHT_ZS]) & (df[WEIGHT_ZS] < h), outcome].tolist()

  fig, ax = plt.subplots(figsize=(15, 10))
  if violin:
    pass
  else:
    ax.boxplot([wz2los[i] for i in wz2los.keys()], widths=0.7, notch=True, patch_artist=True)
  ax.set_title(f"{outcome.replace('_', ' ')} Distribution by Weight z-score Group", fontsize=19, y=1.01)
  ax.set_xlabel("Weight z-score", fontsize=16)
  ax.set_ylabel(f"{outcome.replace('_', ' ')}", fontsize=16)
  labels = [f'{weightz_bins[i]}-{weightz_bins[i + 1]}\n$n$={len(wz2los[i])}' for i in range(len(weightz_bins) - 2)]
  #labels.append(f'{age_bins[-2]}+\n$n$={len(age2los[max(age2los.keys())])}')
  labels = [str(k) for k in wz2los.keys()]
  labels[0] = f'<{weightz_bins[1]}'
  labels[-1] = f'{weightz_bins[-2]}+'
  ax.set_xticklabels(labels, fontsize=13)
  if ylim:
    ax.set_ylim([0, ylim])

  plt.show()


# ---------------------------------------- Language-Interpreter & NNT ----------------------------------------
def language_interpreter_eda(dashb_df, outcome=NNT, preprocess_y=True, freq_range='all', interpreter_cat=False):
  df = dashb_df.copy()
  if interpreter_cat == True:
    df.loc[df[INTERPRETER] == 'Y', LANGUAGE] = 'Foreign & Need Interpreter'
    print('Interpreter Need value count: ', df.groupby(INTERPRETER).size().to_dict())

  if interpreter_cat != 'bipart':
    language_interpreter_eda_violinplot(df, outcome, preprocess_y, freq_range)
  else:
    language_interpreter_eda_violinplot(
      df, outcome, preprocess_y, freq_range,
      f'Interpreter-needed Language vs {outcome} -- frequency ranking between {freq_range}', True)


# Helper function for plotting
def language_interpreter_eda_violinplot(df, outcome, preprocess_y, freq_range, title=None, bipart=False):
  lang2cnt = df.groupby(LANGUAGE).size().to_dict()
  # print('Interpreter Need value count: ', df.groupby(INTERPRETER).size().to_dict())
  print('Language set (#languages = %d): ' % len(lang2cnt), lang2cnt.keys())
  lang_cnt_sorted = sorted(lang2cnt.items(), key=lambda x: x[1], reverse=True)
  x_order = np.array([x[0] for x in lang_cnt_sorted])
  if freq_range != 'all':
    x_order = x_order[freq_range[0]:freq_range[1]]
    df = df[df[LANGUAGE].isin(x_order)]
  y = df[outcome]
  if preprocess_y:
    y[y > MAX_NNT] = MAX_NNT + 1
    df[outcome] = y
  hue, hue_order, split, palette, xticklabels = None, None, False, None, [f'{x} ({lang2cnt[x]})' for x in x_order]
  if bipart:
    df = df[df[INTERPRETER].isin(['N', 'Y'])]
    df.loc[(df[INTERPRETER] == 'N'), INTERPRETER] = False
    df.loc[(df[INTERPRETER] == 'Y'), INTERPRETER] = True
    hue, hue_order, split, palette = INTERPRETER, [True, False], True, {True: 'cornflowerblue', False: 'salmon'}
    lang_cnt_sorted_interTrue = defaultdict(int)
    lang_cnt_sorted_interTrue.update(df[df[INTERPRETER]].groupby(LANGUAGE).size().to_dict())
    xticklabels = [f'{x} {lang_cnt_sorted_interTrue[x], lang2cnt[x]-lang_cnt_sorted_interTrue[x]}'
                   for x in x_order]

  fig, ax = plt.subplots(1, 1, figsize=(16, 9))
  sns.violinplot(x=LANGUAGE, data=df, y=outcome, hue=hue, hue_order=hue_order, split=split, ax=ax,
                 order=x_order, palette=palette)
  ax.set_xlabel('Language (with count)', fontsize=16)
  ax.set_ylabel('Number of Nights', fontsize=16)
  title = f'Language vs {NNT} -- frequency ranking between {freq_range}' if title is None else title
  ax.set_title(title, fontsize=18, y=1.01)
  ax.set_xticklabels(xticklabels, fontsize=13)
  ax.yaxis.set_tick_params(labelsize=13)
  ax.set_ylim([-0.9, 6.9])
  ax.set_xlim([-0.9, 16.4])
  fig.autofmt_xdate(rotation=45)
  plt.show()


def interpreter_eda(dashb_df, outcome=NNT):
  interp2cnt = dashb_df.groupby(INTERPRETER).size.to_dict()
  print('Interpreter_need type count: ', interp2cnt)

  fig, ax = plt.subplots(1, 1, figsize=(12, 9))
  sns.violinplot(data=dashb_df, x=INTERPRETER, y=outcome, ax=ax)
  ax.set_xlabel('Interpreter Flag (with count)', fontsize=16)
  ax.set_ylabel('Number of Nights', fontsize=16)
  ax.set_title(f'Interpreter Flag vs {NNT}', fontsize=18, y=1.01)
  ax.set_ylim([-1, 7])
  ax.yaxis.set_tick_params(labelsize=13)
  ax.set_xticklabels([f'{k} (n={v})' for k, v in interp2cnt.items()], fontsize=13)
  plt.show()


# ---------------------------------------- Major Region & NNT ----------------------------------------
# Major region VS NNT boxplot
def major_region_eda(dashb_df, outcome=NNT, preprocess_y=True):
  y = dashb_df[outcome].to_numpy()
  if preprocess_y:
    y[y > MAX_NNT] = MAX_NNT + 1

  region2cnt = dashb_df[[REGION]].groupby(REGION).size().to_dict()
  region_type = ['Local', 'Regional', 'National', 'International', 'Unknown']

  fig, ax = plt.subplots(1, 1, figsize=(12, 8))
  sns.violinplot(x=dashb_df[REGION], y=y, order=region_type)  # , scale='width'
  ax.set_xlabel('Major Region', fontsize=16)
  ax.set_ylabel('Number of Nights', fontsize=16)
  ax.set_title('NNT vs. Major Region', fontsize=18, y=1.01)
  ax.set_ylim([-0.5, 6.5])
  ax.set_xticklabels([f'{k}\n$n$={region2cnt[k]}' for k in region_type], fontsize=13)
  plt.show()


# ---------------------------------------- State Code & NNT ----------------------------------------
def state_code_eda(dashb_df, outcome=NNT, preprocess_y=True, freq_range='all'):
  # print('State code count: ', dashb_df.groupby(STATE).size().to_dict())
  df = dashb_df.copy()  # df = dashb_df[dashb_df[STATE] != '0']
  state2cnt = df.groupby(STATE).size().to_dict()
  state_cnt_sorted = sorted(state2cnt.items(), key=lambda x: x[1], reverse=True)
  x_order = np.array([x[0] for x in state_cnt_sorted])
  # print('State code set (#codes = %d): ' % len(state2cnt), state2cnt.keys())
  if freq_range != 'all':
    x_order = x_order[freq_range[0]:freq_range[1]]
    df = df.loc[df[STATE].isin([x for x in x_order])]
  y = df[outcome].to_numpy()
  if preprocess_y:
    y[y > MAX_NNT] = MAX_NNT + 1
    df[outcome] = y

  fig, ax = plt.subplots(1, 1, figsize=(18, 9))
  sns.violinplot(data=df, x=STATE, y=outcome, order=x_order)
  ax.set_xlabel('State Code (with count)', fontsize=16)
  ax.set_ylabel('Number of Nights', fontsize=16)
  ax.set_title(f'State Code vs {NNT} -- frequency ranking between {freq_range}', fontsize=18, y=1.01)
  ax.set_ylim([-1, 7])
  ax.set_xticklabels([f'{k} ({state2cnt[k]})' for k in x_order], fontsize=13)
  ax.yaxis.set_tick_params(labelsize=13)
  fig.autofmt_xdate(rotation=45)
  plt.show()


# ---------------------------------------- Miles Traveled & NNT ----------------------------------------
def miles_traveled_eda(dashb_df, q=15, outcome=NNT, preprocess_y=True, violin=True):
  fig, ax = plt.subplots(1, 1, figsize=(20, 9))
  Xdf = dashb_df[[MILES, outcome]]
  Xdf = Xdf[Xdf[MILES].notnull()]
  qs = pd.qcut(Xdf[MILES], q=q)
  qs = [round(qval.right, 1) for qval in qs.values]
  Xdf[MILES] = qs
  #ax.set_xlim([0, 1000])
  ax.set_ylim([-1, 40])
  outy = Xdf[outcome].to_numpy()
  if preprocess_y:
    outy[outy > MAX_NNT] = MAX_NNT + 1
    Xdf[outcome] = outy
    ax.set_ylim([-0.6, 6.6])

  if violin:
    sns.violinplot(data=Xdf, x=MILES, y=outcome)
    #ax.set_xlim([-20, 500])
  else:
    ax.scatter(dashb_df[MILES], outy)
  ax.set_xlabel(f'Miles traveled ({q}-quantile upperbound)', fontsize=16)
  ax.set_ylabel('Number of Bed Nights', fontsize=16)
  ax.set_title('NNT vs. Miles traveled', fontsize=18, y=1.01)
  ax.yaxis.set_tick_params(labelsize=13)

  plt.show()


# ---------------------------------------- Organ System Code ----------------------------------------
# Surgical case histogram VS OS code
def organ_system_eda(dashboard_df):
  diag_cnts = [dashboard_df[lbl].sum() for lbl in diaglabels]

  figs, axs = plt.subplots(1, 2, figsize=(18, 7))
  cmap = plt.cm.tab20
  colors = cmap(np.linspace(0., 1., len(diaglabels)))
  axs[0].pie(diag_cnts, labels=diaglabels, colors=colors, autopct='%.2f%%', startangle=90,
             wedgeprops={"edgecolor": "gray", 'linewidth': 0.3})
  axs[0].set_title("Organ System Diagnosis Code Distribution", y=1.07, fontsize=15)
  axs[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

  dashboard_df['OS_code_count'] = dashboard_df[diaglabels].sum(axis=1)
  print("Number of patients with more than 1 diagnosed conditions: ",
        dashboard_df[dashboard_df['OS_code_count'] > 1].shape[0])

  bins = range(0, 22, 1)
  arr = axs[1].hist(dashboard_df['OS_code_count'], bins=bins,
              color="purple", alpha=0.6, edgecolor="black", linewidth=0.5)
  axs[1].set_title("Number of Organ System Codes Distribution", fontsize=15,
                   y=1.05)
  axs[1].set_xlabel("Number of organ system codes")
  axs[1].set_ylabel("Number of patients")
  for i in bins[:-1]:
    plt.text(arr[1][i],arr[0][i],str(int(arr[0][i])))
  plt.show()


# Organ system code & LOS
def organ_system_los_boxplot(dashboard_df, exclude_outliers=0, xlim=None):
  # Box plot of LoS distribution over all 21 OS codes
  #dashboard_df = utils.drop_outliers(dashboard_df, exclude_outliers)

  diag2los = defaultdict(list)
  for diag in diaglabels:
    diag2los[diag] = dashboard_df[dashboard_df[diag] == 1].LENGTH_OF_STAY.tolist()
  dlabels = [k for k, _ in sorted(diag2los.items(), key=lambda kv: np.median(kv[1]))]

  fig, ax = plt.subplots(figsize=(25, 17))
  cmap = plt.cm.get_cmap('tab20')
  colors = [cmap(i) for i in range(len(dlabels))]
  colors[-1] = plt.cm.get_cmap('tab20b')(2)

  bplot = ax.boxplot([diag2los[d] for d in dlabels], widths=0.7, notch=True, vert=False,
                     patch_artist=True, flierprops={'color': colors})
  ax.set_title("LoS Distribution by Organ System Diagnosis Code", fontsize=22, y=1.01)
  ax.set_xlabel("LoS (day)", fontsize=16)
  ax.set_yticklabels(dlabels, fontsize=14)
  ax.set_ylabel("Organ System Code", fontsize=16)
  if xlim:
    ax.set_xlim([0, xlim])

  for patch, color in zip(bplot["fliers"], colors):
    patch.set_markeredgecolor(color)
    patch.set_markeredgewidth(2)

  for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)


# --------------------------------------- Max total CHEWS & LOS ---------------------------------------
# TODO: How and when is the score generated?
def chews_los_eda(dashboard_df, ylim=None):
  max_chew, min_chew = int(np.ceil(max(dashboard_df.MAX_TOTAL_CHEWS) + DELTA)), int(np.floor(min(dashboard_df.AGE_AT_PROC_YRS)))
  chew_bins = np.linspace(min_chew, max_chew, int(np.ceil(max_chew - min_chew)))

  figs, axs = plt.subplots(1, 1, figsize=(12, 9))
  axs.hist(dashboard_df.MAX_TOTAL_CHEWS, bins=chew_bins, alpha=0.7, edgecolor="black", linewidth=0.5)
  axs.set_title("Max Total CHEWs distribution", fontsize=15)
  axs.set_xlabel("Max Total CHEWs")
  axs.set_ylabel("Number of patients")

  # # Average and median LoS of each age group
  chew2los = []
  avglos_chew = [0] * len(chew_bins)
  medlos_chew = [0] * len(chew_bins)
  print(max_chew)
  for c in range(max_chew):
    los = dashboard_df[c == dashboard_df.MAX_TOTAL_CHEWS].LENGTH_OF_STAY
    avglos_chew[c] = np.mean(los)
    medlos_chew[c] = np.median(los)
    chew2los.append(los)

  figs, axs = plt.subplots(1, 1, figsize=(12, 8))
  cmap = plt.cm.tab20
  colors = [cmap(i) for i in range(len(chew2los))]
  bplot = axs.boxplot(chew2los, widths=0.7, notch=True, vert=True,
                      patch_artist=True)
  # axs[1].bar(chew_bins, avglos_chew, alpha=0.5, color="red", edgecolor="black", linewidth=0.5)
  axs.set_title("LoS Distribution for each CHEWs Count", fontsize=15)
  axs.set_xlabel("Max Total CHEWs", fontsize=13)
  axs.set_ylabel("LoS", fontsize=13)
  if ylim:
    axs.set_ylim([0, ylim])

  # bplot = ax.boxplot([diag2los[d] for d in diaglabels], widths=0.7, notch=True, vert=False,
  #                    patch_artist=True,flierprops={'color':colors})
  # ax.set_title("LoS Distribution by Organ System Diagnosis Code", fontsize=22, y=1.01)
  # ax.set_xlabel("LoS (day)", fontsize=16)
  # ax.set_yticklabels(diaglabels, fontsize=14)
  # ax.set_ylabel("Organ System Code", fontsize=16)

  for patch, color in zip(bplot["fliers"], colors):
    patch.set_markeredgecolor(color)
    patch.set_markeredgewidth(2)

  for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

