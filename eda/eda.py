"""
EDA on non-diagnosis features
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .globals import diaglabels, DELTA
from . import plot_utils
from . import utils


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

