"""
EDA on non-diagnosis features
"""

from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..globals import diaglabels, DELTA
from .. import globals
from ..c1_data_preprocessing import preprocess_y
from .. import utils_plot
from .. import utils


# -------------------------------------------- LOS --------------------------------------------
def los_histogram(y, dataType='Training', ax=None):
  if ax is None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
  outcome_cnter = Counter(y)
  ax.barh(range(globals.MAX_NNT + 2), [outcome_cnter[i] for i in range(globals.MAX_NNT + 2)], align='center')
  ax.set_xlabel("Number of surgical cases")
  ax.invert_yaxis()
  ax.set_yticks(range(globals.MAX_NNT + 2))
  ax.set_yticklabels(globals.NNT_CLASS_LABELS, fontsize=13)
  ax.set_title("LoS Histogram (%s)" % dataType)
  rects = ax.patches
  total_cnt = len(y)
  labels = ["{:.1%}".format(outcome_cnter[i] / total_cnt) for i in range(globals.MAX_NNT + 2)]
  for rect, label in zip(rects, labels):
    ht, wd = rect.get_height(), rect.get_width()
    ax.text(wd + 2.5, rect.get_y() + ht / 2, label,
                ha='left', va='center', fontsize=15)



# ---------------------------------------- Gender & LOS ----------------------------------------
def gender_eda(dashboard_df):
  labels = ["Male", "Female"]
  gender_cnts = [dashboard_df[dashboard_df.SEX_CODE == "M"].shape[0],
                 dashboard_df[dashboard_df.SEX_CODE == "F"].shape[0]]

  figs, axs = plt.subplots(1, 2, figsize=(18, 6))
  axs[0].pie(gender_cnts, labels=labels, autopct='%.2f%%', startangle=90)
  axs[0].set_title("Gender distribution", fontsize=15)
  axs[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

  bins = np.linspace(0, 21, 40)
  axs[1].set_xlim([bins[0], bins[-1]])
  axs[1].hist([dashboard_df[dashboard_df.SEX_CODE == "M"].LENGTH_OF_STAY,
               dashboard_df[dashboard_df.SEX_CODE == "F"].LENGTH_OF_STAY],
              bins, alpha=0.7, edgecolor='black', linewidth=0.5, label=["Male", "Female"])
  axs[1].set_title("LoS Distribution (Male vs Female)", fontsize=15)
  axs[1].set_xlabel("Length of Stay (day)")
  axs[1].set_ylabel("Number of patients")

  plt.legend()
  plt.show()

# ---------------------------------------- Language-Interpreter & NNT ----------------------------------------
def language_interpreter_eda(dashb_df, outcome=globals.NNT, preprocess_y=True, topK_freq='all', interpreter_cat=False):
  df = dashb_df.copy()
  if interpreter_cat == True:
    df.loc[df[globals.INTERPRETER] == 'Y', globals.LANGUAGE] = 'Foreign & Need Interpreter'
    print('Interpreter Need value count: ', df.groupby(globals.INTERPRETER).size().to_dict())
  language_interpreter_eda_violinplot(df, outcome, topK_freq)

  if interpreter_cat == 'separate':
    print('Interpreter Need value count: ', df.groupby(globals.INTERPRETER).size().to_dict())
    df = dashb_df.loc[dashb_df[globals.INTERPRETER] == 'Y']
    language_interpreter_eda_violinplot(df, outcome, topK_freq, f'Interpreter-needed Language vs {outcome}')


# Helper function for plotting
def language_interpreter_eda_violinplot(df, outcome, topK_freq, title=None):
  lang2cnt = df.groupby(globals.LANGUAGE).size().to_dict()
  lang_cnt_sorted = sorted(lang2cnt.items(), key=lambda x: x[1], reverse=True)
  lang_col = df[globals.LANGUAGE]
  if topK_freq != 'all':
    lang_cnt_sorted = lang_cnt_sorted[:topK_freq]
    lang_df = df.loc[df[globals.LANGUAGE].isin([x[0] for x in lang_cnt_sorted])]
    lang_col, y = lang_df[globals.LANGUAGE], lang_df[outcome]
  else:
    y = df[outcome]
  if preprocess_y:
    y[y > globals.MAX_NNT] = globals.MAX_NNT + 1

  # English, Spanish, ...
  fig, ax = plt.subplots(1, 1, figsize=(16, 9))
  sns.violinplot(lang_col, y, ax=ax, order=[x[0] for x in lang_cnt_sorted])
  ax.set_xlabel('Language (with count)', fontsize=16)
  ax.set_ylabel('Number of Nights', fontsize=16)
  title = f'Language vs {globals.NNT}' if title is None else title
  ax.set_title(title, fontsize=18, y=1.01)
  ax.set_xticklabels([f'{x[0]} ({x[1]})' for x in lang_cnt_sorted], fontsize=13)
  ax.yaxis.set_tick_params(labelsize=13)
  fig.autofmt_xdate(rotation=45)
  plt.show()


# ---------------------------------------- Major Region & NNT ----------------------------------------
# Major region VS NNT boxplot
def major_region_eda(dashb_df, outcome=globals.NNT, preprocess_y=True):
  y = dashb_df[outcome].to_numpy()
  if preprocess_y:
    y[y > globals.MAX_NNT] = globals.MAX_NNT + 1

  region2cnt = dashb_df[[globals.REGION]].groupby(globals.REGION).size().to_dict()
  region_type = [k for k in region2cnt.keys()]
  cnt_pct = [region2cnt[k] / dashb_df.shape[0] for k in region_type]

  fig, ax = plt.subplots(1, 1, figsize=(12, 8))
  sns.violinplot(dashb_df[globals.REGION], y, order=region_type, scale='width')  # , widths=cnt_pct
  ax.set_xlabel('Major Region', fontsize=16)
  ax.set_ylabel('Number of Nights', fontsize=16)
  ax.set_title('NNT vs. Major Region', fontsize=18, y=1.01)
  ax.set_ylim([-1, 7])
  ax.set_xticklabels([f'{k}\n$n$={region2cnt[k]}' for k in region_type], fontsize=13)
  plt.show()


# ---------------------------------------- Miles Traveled & NNT ----------------------------------------
def miles_traveled_eda(dashb_df, outcome=globals.NNT, preprocess_y=True, violin=True):
  fig, ax = plt.subplots(1, 1, figsize=(12, 9))
  ax.set_xlim([0, 1000])
  ax.set_ylim([-1, 40])
  outy = dashb_df[outcome]
  if preprocess_y:
    outy[outy > globals.MAX_NNT] = globals.MAX_NNT + 1
    ax.set_ylim([-1, 10])

  if violin:
    sns.violinplot(y=outy, x=dashb_df[globals.MILES], orient='h')
    ax.set_xlim([-20, 500])
  else:
    ax.scatter(dashb_df[globals.MILES], outy)
  ax.set_xlabel('Miles traveled', fontsize=16)
  ax.set_ylabel('Number of Bed Nights', fontsize=16)
  ax.set_title('NNT vs. Miles traveled', fontsize=18)

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


# ---------------------------------------- Age & LOS ----------------------------------------
def age_los_eda(dashboard_df):
  max_age, min_age = np.ceil(max(dashboard_df.AGE_AT_PROC_YRS) + DELTA), np.floor(min(dashboard_df.AGE_AT_PROC_YRS))
  age_bins = np.linspace(int(min_age), int(max_age), int(np.ceil(max_age - min_age)))

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


def age_los_boxplot(df, max_age=30, bin_width=1, ylim=None):
  age2los = defaultdict()
  last_i = 0
  for i in range(bin_width, max_age + 1, bin_width):
    age2los[i] = df[(i - bin_width <= df.AGE_AT_PROC_YRS) & (df.AGE_AT_PROC_YRS < i)].LENGTH_OF_STAY.tolist()
    last_i = i
  age2los[last_i] = df[last_i <= df.AGE_AT_PROC_YRS].LENGTH_OF_STAY.tolist()

  fig, ax = plt.subplots(figsize=(15, 10))
  bplot = ax.boxplot([age2los[i] for i in age2los.keys()], widths=0.7, notch=True, patch_artist=True)
  ax.set_title("LoS Distribution by Age Group", fontsize=19, y=1.01)
  ax.set_xlabel("Age (yr)", fontsize=16)
  ax.set_ylabel("LoS (day)", fontsize=16)
  labels = [str(k) for k in age2los.keys()]
  labels[-1] = str(last_i) + "+"
  ax.set_xticklabels(labels)
  if ylim:
    ax.set_ylim([0, ylim])

  plt.show()


# ---------------------------------------- Weight z-score & LOS ----------------------------------------
#
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


def weightz_los_boxplot(df, weightz_range, bin_width=1, ylim=None):
  wz2los = defaultdict()
  wz2los[weightz_range[0]-1] = df[df.WEIGHT_ZSCORE < weightz_range[0]].LENGTH_OF_STAY.tolist()
  last_i = 0
  for i in range(weightz_range[0], weightz_range[-1] + 1, bin_width):
    wz2los[i] = df[(i - bin_width <= df.WEIGHT_ZSCORE) & (df.WEIGHT_ZSCORE < i)].LENGTH_OF_STAY.tolist()
    last_i = i
  wz2los[last_i] = df[last_i <= df.WEIGHT_ZSCORE].LENGTH_OF_STAY.tolist()

  fig, ax = plt.subplots(figsize=(15, 10))
  bplot = ax.boxplot([wz2los[i] for i in wz2los.keys()], widths=0.7, notch=True, patch_artist=True)
  ax.set_title("LoS Distribution by Weight z-score Group", fontsize=19, y=1.01)
  ax.set_xlabel("Weight z-score", fontsize=16)
  ax.set_ylabel("LoS (day)", fontsize=16)
  labels = [str(k) for k in wz2los.keys()]
  labels[0] = '<' + str(weightz_range[0])
  labels[-1] = str(last_i) + "+"
  ax.set_xticklabels(labels)
  if ylim:
    ax.set_ylim([0, ylim])

  plt.show()

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

