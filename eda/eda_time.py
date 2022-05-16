"""
EDA on date-time related features,
such as time of admission, discharge, end of surgery, actual start (wheel-in), actual stop (wheel out) etc.
"""
import datetime
import random

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter

from globals import *
from globals_fs import *
from c1_data_preprocessing import gen_y_nnt


Event2DatetimeCol = {ADMIT: 'HAR_ADMIT_DATE', DISCHARGE: 'HAR_DISCHARGE_DATE',
                     SURGEONSTART: 'SURGEON_START_DT_TM', SURGEONEND: 'SURGERY_END_DT_TM'}


# -------------------------------- Patient hospital event time of day & NNT --------------------------------
def plot_event_time_of_day_los(dashb_df, event=ADMIT, outcome=NNT, unit=HOUR, save=False):
  event_col = Event2DatetimeCol.get(event, None)
  if event_col is None:
    raise ValueError(f'{event} column is not available yet!')

  df = dashb_df[[event_col, outcome]]
  if unit == HOUR:
    unit_col = f'{event_col}_unit'
    unit_label = 'Hour of Day'
    df[unit_col] = df[event_col].dt.hour
  elif unit == DAY:
    unit_col = f'{event_col}_unit'
    unit_label = 'Day of Week'
  else:
    raise NotImplementedError

  hr2cnt = df.groupby(by=unit_col).size().to_dict()
  print(hr2cnt)

  fig, ax = plt.subplots(1, 1, figsize=(15, 10))
  if outcome == NNT:
    ax.set_yticklabels(NNT_CLASS_LABELS)
    df.loc[df[outcome] > MAX_NNT] = MAX_NNT + 1
  sns.violinplot(data=df, x=unit_col, y=outcome, ax=ax)
  ax.set_title(f'{event} {unit_label} & {outcome} Distribution', fontsize=19)
  ax.set_xlabel(unit_label.lower().capitalize(), fontsize=16)
  ax.set_ylabel(outcome, fontsize=16)
  if save:
    plt.savefig(FIG_DIR / f'{event}-{unit_label}.png', dpi=200)


# ---------------------------------- Patient hospital event & time of day ----------------------------------
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
  new_cmap = colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
    cmap(np.linspace(minval, maxval, n)))
  return new_cmap


# histograms (admit, discharge, end of surgery, wheel-out time ...)
def event_historgram_by_time(df, event='ADMIT', unit=HOUR, save=False):
  total = df.SURG_CASE_KEY.nunique()
  alpha = 0.9
  if event == ADMIT_DTM:
    event_txt = "Admission"
    #alpha = 0.65
  elif event == WHEELOUT:
    #event_df = df.groupby(df.ACTUAL_STOP_DATE_TIME.dt.hour).SURG_CASE_KEY.count()
    event_txt = "Wheel-out"
  elif event == DISCHARGE_DTM:
    #event_df = df.groupby(df[DISCHARGE_DTM].dt.hour).SURG_CASE_KEY.count()
    event_txt = "Discharge"
  elif event == IPSTART:
    #event_df = df.groupby(df.INCISION_PROCEDURE_START_DATE_TIME.dt.hour).SURG_CASE_KEY.count()
    event_txt = "Incision Procedure Start"
  elif event == SURG_START_DTM:
    #event_df = df.groupby(df[SURG_START_DTM].dt.hour).SURG_CASE_KEY.count()
    event_txt = "Surgeon Start"
  elif event == SURG_END_DTM:
    #event_df = df.groupby(df[SURG_END_DTM].dt.hour).SURG_CASE_KEY.count()
    event_txt = "Surgery End"
  else:
    raise KeyError("%s is not supported for EDA" % event)
  df = df[[event, 'SURG_CASE_KEY', NNT]]
  df[NNT] = gen_y_nnt(df[NNT])
  if unit == HOUR:
    # event_df = df.groupby(df[event].dt.hour)['SURG_CASE_KEY'].count()
    # for h in range(24):
    #   if event_df.get(h) is None:
    #     event_df[h] = 0
    # event_cnts = [event_df[i] for i in range(24)]
    event_df = df.groupby(by=[df[event].dt.hour, NNT]).size().unstack(-1)
    #event_df.columns = ['$\leq$1', '2', '3', '4', '5', '$\geq$6']
    event_df.columns = ['LOS$\leq$1', 'LOS = 2', 'LOS = 3', 'LOS = 4', 'LOS = 5', 'LOS$\geq$6']

    event_df = event_df.fillna(0).reset_index()
    unit_label = 'Hour of Day'
    xs = np.arange(24)
    i = event_df.shape[0]
    for x in xs:
      if x not in event_df[event].to_list():
        event_df.loc[i] = [x] + [0] * NNT_CLASS_CNT
        i += 1
    event_df = event_df.sort_values(by=event)
    xticklabels = list(map(int, range(0, 23)))
  elif unit == DAY:
    # event_df = df.groupby(df[event].dt.dayofweek)['SURG_CASE_KEY'].count()
    # for h in range(7):  # 0 - Monday, 6 - Sunday
    #   if event_df.get(h) is None:
    #     event_df[h] = 0
    # event_cnts = [event_df[i] for i in range(7)]
    event_df = df.groupby(by=[df[event].dt.dayofweek, NNT]).size().unstack(-1)
    event_df.columns = ['LOS$\leq$1', 'LOS=2', 'LOS=3', 'LOS=4', 'LOS=5', 'LOS$\geq$6']
    event_df = event_df.reset_index()
    unit_label = 'Day of Week'
    xs = np.arange(7)
    xticklabels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
  else:
    raise NotImplementedError

  fig, ax = plt.subplots(1, 1, figsize=(12, 9))
  #ax.bar(xs, event_cnts, width=0.76)
  event_df.plot(x=event, kind='bar', stacked=True, rot=0, ax=ax, width=0.76,
                colormap=truncate_colormap(plt.get_cmap('coolwarm'), 0.1, 0.95, NNT_CLASS_CNT))
  if event == DISCHARGE_DTM and unit == HOUR:
    ax.axvline(x=10.5, ls='--', lw=1.5, color='red', alpha=0.9, label='Time of Discharge Notice')
    ax.legend(prop={'size': 13})
  else:
    ax.legend(title='LOS Outcome Class')

  ax.set_xticks(xs if unit == DAY else xs - 0.5)
  ax.set_xticklabels(xticklabels, fontsize=13 if unit == HOUR else 14)
  ax.set_title(f"%s {unit_label} Histogram" % event_txt, fontsize=19, y=1.01)
  ax.set_xlabel(f"%s {unit_label.lower()}" % event_txt, fontsize=16, labelpad=15)
  ax.set_ylabel("Number of cases", fontsize=16)
  ax.yaxis.set_tick_params(labelsize=13)

  rects = ax.patches
  event_counter = event_df.set_index(event).sum(axis=1).sort_values()
  for i in xs:
    rect = rects[i]
    if event_counter[i] > 0:
      x, wd = rect.get_x(), rect.get_width()
      ht = event_counter[i]
      ax.text(x + wd / 2, ht + 20, "{:.1%}".format(event_counter[i] / total),
              ha='center', va='bottom', fontsize=8 if unit == HOUR else 12)
  if save:
    plt.savefig(FIG_DIR / f'{event}_histogram-{unit_label}.png', dpi=200)
  return event_df


# ---------------------------------- Histogram of Pre-admission Wait Time ----------------------------------
def pre_admission_wait_time_hist(df, bins=None):
  df = df[[ADMIT_DTM, BOOKING_DTM]].apply(pd.to_datetime)
  df['pre-admission_wait_days'] = (df[ADMIT_DTM] - df[BOOKING_DTM]) / np.timedelta64(1, 'D')
  print('Minimum pre-admission wait days: ', df['pre-admission_wait_days'].min())
  fig, ax = plt.subplots(1, 1, figsize=(12, 8))
  if bins is None:
    ax.hist(df['pre-admission_wait_days'], bins=20)
  else:
    if not np.isinf(bins[-1]):
      bins = np.concatenate([bins, [float('inf')]])
    counts, edges = np.histogram(df['pre-admission_wait_days'], bins)
    ax.bar(edges[:-1], counts, alpha=0.8, color='gray', align='center')
    labels = list(map(str, edges[:-1]))
    labels[-1] = f'{labels[-1]}+'
    ax.set_xticks(edges[:-1])
    ax.set_xticklabels(labels, fontsize=13)
    print(counts, '\n', edges)

  ax.set_xlabel('Pre-admission Wait Days', fontsize=14)
  ax.set_ylabel('Number of Cases', fontsize=14)
  ax.set_title('Pre-admission Wait Days Histogram', fontsize=16, y=1.01)


# -----------------------------------------  Postoperative LoS Histogram -----------------------------------------
# Calculate post-operative LoS
def gen_postop_los(df, counting_unit="H"):
  post_los_df = (df[DISCHARGE_DTM] - df[SURG_END_DTM]).astype('timedelta64[ns]')
  # print("before rounding", post_los_df.head(6))
  if counting_unit == HOUR:
    post_los_df = post_los_df.dt.round('1h').astype('timedelta64[h]')
    unit_txt = "hour"
    # print("after rounding: ", post_los_df.head(6))
  elif counting_unit == DAY:
    post_los_df = post_los_df.dt.round('1h').astype('timedelta64[D]')
    unit_txt = "day"
  elif counting_unit == NIGHT:
    post_los_df = (df[DISCHARGE_DTM].dt.date - df[SURG_END_DTM].dt.date) / np.timedelta64(1, 'D')
    unit_txt = "night"
  else:
    raise ValueError("Unit %d is not supported!" % counting_unit)
  return post_los_df, unit_txt


def postop_los_histogram(df, unit="H", exclude_outliers=0, plot_los=False, ):
  post_los_df, unit_txt = gen_postop_los(df, unit)
  min_postLos, max_postLos = int(min(post_los_df)), int(max(post_los_df))
  print("Min: ", min_postLos, "; Max: ", max_postLos)

  if exclude_outliers > 0:
    post_los_df.drop(post_los_df.nlargest(exclude_outliers).index, inplace=True)
    max_postLos = int(max(post_los_df))

  # Plot
  fig, ax = plt.subplots(figsize=(14, 7))
  bins = np.linspace(min_postLos, max_postLos + 0.5, 100)
  ax.set_xlim([bins[0], bins[-1]])

  if plot_los:
    los_df = df.LENGTH_OF_STAY if unit == DAY else df.LENGTH_OF_STAY * 24
    if exclude_outliers > 0:
      los_df.drop(los_df.nlargest(exclude_outliers).index, inplace=True)

    ax.hist([post_los_df, los_df], bins, edgecolor='black', alpha=0.8, linewidth=0.5,
            label=["Postoperative LoS", "LoS"])
  else:
    ax.hist(post_los_df, bins, color="orange", edgecolor='black', alpha=1, linewidth=0.5)

  plt.title("Los & Postoperative LoS Histogram" if plot_los else "Postoperative LoS Histogram")
  plt.xlabel("LoS (%s)" % unit_txt)
  plt.ylabel("Number of surgical cases")
  plt.legend()
  plt.show()

  return post_los_df


def los_postop_los_scatter(df, unit=HOUR, xylim=None):
  postop_los_df, unit_txt = gen_postop_los(df, counting_unit=unit)
  los_df = df.LENGTH_OF_STAY * 24 if unit == HOUR else df.LENGTH_OF_STAY

  fig, ax = plt.subplots(figsize=(11,10))
  ax.scatter(los_df, postop_los_df, s=20, facecolors='none', edgecolors='purple')
  ax.set_title("LOS VS Postoperative LOS (%s)" % unit_txt, fontsize=16)
  ax.set_xlabel("Length of stay (%s)" % unit_txt, fontsize=14)
  ax.set_ylabel("Postoperative LOS (%s)" % unit_txt, fontsize=14)
  if xylim:
    ax.set_xlim([0, xylim])
    ax.set_ylim([0, xylim])
  lims = np.array([np.min([ax.get_xlim(), ax.get_ylim()]),
                   np.max([ax.get_xlim(), ax.get_ylim()])])
  ax.plot(lims, lims, '--', color='k', linewidth=3, alpha=0.8)
  if unit == HOUR:
    ax.plot(lims, lims - 10, '--', color='r', linewidth=2.5, alpha=0.8)

  plt.show()

# -----------------------------------------  Operative Length and Post-op LoS -----------------------------------------
def op_length_postopLos_eda(dashb_df, max_los=5, xlim=None, ylim=None, preprocess_los=False, save=False):
  #fig, ax = plt.subplots(figsize=(12, 9))
  figs, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 1]})
  ax, ax2 = axs[0], axs[1]
  df = dashb_df[[LOS, SURG_START_DTM, SURG_END_DTM, DISCHARGE_DTM]]
  df['op_length'] = (pd.to_datetime(dashb_df[SURG_END_DTM]) - pd.to_datetime(dashb_df[SURG_START_DTM]))\
                    / np.timedelta64(1, 'h')
  df['surg_end_elapsed_time'] = (pd.to_datetime(dashb_df[SURG_END_DTM]) - pd.to_datetime(dashb_df[ADMIT_DTM]))\
                                / np.timedelta64(1, 'h')
  df['postop_los'] = (pd.to_datetime(dashb_df[DISCHARGE_DTM]).dt.date - pd.to_datetime(dashb_df[SURG_END_DTM]).dt.date)\
                     / np.timedelta64(1, 'D')
  if max_los:
    df['postop_los'] = df['postop_los'].apply(lambda x: max_los + 1 if x > max_los else int(np.round(x)))
  if preprocess_los:
    df['postop_los'] = df['postop_los'].apply(lambda x: max(x, 1) if x <= max_los else max_los+1)
    yticklabels = ['$\leq$ 1'] + list(map(str, np.arange(2, max_los + 1))) + [f'$\geq$ {max_los + 1}']
  else:
    yticklabels = list(map(str, np.arange(0, max_los+1))) + [f'$\geq$ {max_los+1}']
  if xlim:
    df['op_length'] = df['op_length'].apply(lambda x: xlim + random.uniform(0, 0.5) if x >= xlim else x)
    ax.set_xlim([-0.3, xlim + 0.5])
    ax.set_xticks(np.arange(xlim+1))

  #ax.scatter(df['op_length'], df['postop_los'], s=20, facecolors='none', edgecolors='purple')
  cmap = plt.cm.get_cmap('coolwarm')
  colors = cmap(np.linspace(0.1, 0.95, max_los+1))  # max_los+2
  print(Counter(df['postop_los']))
  bplot = ax.boxplot([df.loc[df['postop_los'] == x, 'op_length'].to_list() for x in np.arange(1, max_los+2)], widths=0.7,
                     notch=True, vert=False, patch_artist=True, flierprops={'color': colors})
  ax.set_title('Post-operative LOS Outcome over Operative Length', y=1.02, fontsize=22)
  ax.set_xlabel('Operative length (hour)', fontsize=18)
  ax.set_ylabel('Post-operative LOS outcome class', fontsize=18)
  ax.set_yticks(np.arange(1, max_los+4))
  ax.set_yticklabels(yticklabels, fontsize=15)
  ax.xaxis.set_tick_params(labelsize=14)
  ax.set_ylim([0, max_los + 2])
  if ylim:
    ax.set_ylim([0, ylim])

  for patch, color in zip(bplot["fliers"], colors):
    patch.set_markeredgecolor(color)
    patch.set_markeredgewidth(2)

  for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

  for median in bplot['medians']:
    median.set_color('yellow')

  postop_los_counter = Counter(df['postop_los'])
  ax2.barh(np.arange(max_los+2), [postop_los_counter[i] for i in np.arange(max_los+2)], align='center', alpha=0.9,
           height=0.7)
  ax2.set_xlabel("Number of cases", fontsize=18)
  # ax.invert_yaxis()
  ax2.set_yticks(np.arange(1, max_los+4))
  ax2.set_yticklabels([''] * (max_los+1), fontsize=13)
  ax2.set_ylim([0, max_los + 2])

  ax2.set_title("Post-operative LOS Outcome Histogram", fontsize=22, y=1.02)
  rects = ax2.patches
  labels = ["{:.1%}".format(postop_los_counter[i] / df.shape[0]) for i in np.arange(max_los+2)]
  i = 0
  for rect, label in zip(rects, labels):
    if i == 0:
      i += 1
      continue
    ht, wd = rect.get_height(), rect.get_width()
    ax2.text(wd + 50, rect.get_y() + ht / 2, label,
             ha='left', va='center', fontsize=12)
  # ax2.set_xlim([0, 1.1 * max(topK_pproc_dict['case_cnt'].values())])
  # ax2.set_ylim([-0.5, max_los + 2])
  plt.tight_layout(w_pad=-1.5)
  if save:
    plt.savefig(FIG_DIR / 'op_length.png', dpi=200)



# ---------------------------------- Past x-day Readmission Flag - LOS distribution ----------------------------------
def gen_revisit_col(dashb_df, window=30):
  # MRN, SURG_CASE_KEY, DISCHARGE_DTM, ADMIT_DTM
  df = dashb_df.loc[:, ['MRN', 'SURG_CASE_KEY', ADMIT_DTM, DISCHARGE_DTM]]
  df = df.join(df.groupby('MRN').count()['SURG_CASE_KEY'].reset_index(name='visit_cnt').set_index('MRN'), on='MRN', how='left')
  df = df[df['visit_cnt'] > 1]
  surg_key_to_revisit = defaultdict(int)
  for mrn in df['MRN'].unique():
    mrn_df = df.loc[df['MRN'] == mrn, ['SURG_CASE_KEY', ADMIT_DTM, DISCHARGE_DTM]].sort_values(by=ADMIT_DTM)
    prev_discharge_dtm = pd.to_datetime(mrn_df.iloc[0][DISCHARGE_DTM])
    for i in range(1, len(mrn_df)):
      admit_dtm = pd.to_datetime(mrn_df.iloc[i][ADMIT_DTM])
      if (admit_dtm - prev_discharge_dtm) / np.timedelta64(1, 'D') <= window:
        # print(admit_dtm, prev_discharge_dtm)
        surg_key_to_revisit[mrn_df.iloc[i]['SURG_CASE_KEY']] = 1
      prev_discharge_dtm = pd.to_datetime(mrn_df.iloc[i][DISCHARGE_DTM])

  return surg_key_to_revisit


# LOS distribution over Readmission Indicator
def readmission_eda(dashb_df, outcome=NNT, preprocess_y=True, window=30):
  fig, ax = plt.subplots(figsize=(12, 8))
  df = dashb_df[[outcome, 'revisit']]
  if outcome == NNT and preprocess_y:
    df[outcome] = gen_y_nnt(df[outcome])
    ax.set_xticks(sorted(df[outcome].unique()))
    ax.set_xticklabels(NNT_CLASS_LABELS, fontsize=14)
  revisit_counter = Counter(df[dashb_df['revisit'] == 1][outcome])
  no_revisit_counter = Counter(df[dashb_df['revisit'] == 0][outcome])
  revisit_total, non_revisit_total = sum(revisit_counter.values()), sum(no_revisit_counter.values())
  xs = np.array(sorted(no_revisit_counter.keys()))
  ax.bar(xs-0.2, [100 * revisit_counter[x] / revisit_total for x in xs], label=f'{window}-day Readmission', width=0.4,
         alpha=0.8, color='salmon')
  ax.bar(xs+0.2, [100 * no_revisit_counter[x] / non_revisit_total for x in xs], label='Non-readmission', width=0.4,
         alpha=0.8, color='steelblue')
  ax.set_title(f'LOS Distribution by {window}-day Readmission Indicator', fontsize=19, y=1.01)
  ax.set_xlabel('Length of stay', fontsize=16)
  ax.set_ylabel('Case count percentage (%)', fontsize=16)
  ax.yaxis.set_tick_params(labelsize=14)
  ax.legend(prop={'size': 14})

  rects = ax.patches
  labels = [int(revisit_counter[x]) for x in xs]
  for rect, label in zip(rects[:6], labels):
    ht, wd = rect.get_height(), rect.get_width()
    ax.text(rect.get_x() + wd / 2, ht+0.5, label,
            ha='center', va='bottom', fontsize=11)

  labels = [int(no_revisit_counter[x]) for x in xs]
  for rect, label in zip(rects[6:], labels):
    ht, wd = rect.get_height(), rect.get_width()
    ax.text(rect.get_x() + wd / 2, ht + 0.5, label,
            ha='center', va='bottom', fontsize=11)
  plt.savefig(f'readmission{window}.png', dpi=300)
