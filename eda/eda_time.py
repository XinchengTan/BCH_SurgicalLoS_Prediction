"""
EDA on date-time related features,
such as time of admission, discharge, end of surgery, actual start (wheel-in), actual stop (wheel out) etc.
"""
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter

import globals
from globals import *
from c1_data_preprocessing import gen_y_nnt


Event2DatetimeCol = {ADMIT: 'HAR_ADMIT_DATE', DISCHARGE: 'HAR_DISCHARGE_DATE',
                     SURGEONSTART: 'SURGEON_START_DT_TM', SURGEONEND: 'SURGERY_END_DT_TM'}


# -------------------------------- Patient hospital event time of day & NNT --------------------------------
def plot_event_time_of_day_los(dashb_df, event=ADMIT, outcome=NNT):
  event_col = Event2DatetimeCol.get(event, None)
  if event_col is None:
    raise ValueError(f'{event} column is not available yet!')
  df = dashb_df[[event_col, outcome]]
  df[event_col+'_HOUR'] = df[event_col].dt.hour
  hr2cnt = df.groupby(by=event_col+'_HOUR').size().to_dict()
  print(hr2cnt)

  fig, ax = plt.subplots(1, 1, figsize=(15, 10))
  if outcome == NNT:
    ax.set_yticklabels(NNT_CLASS_LABELS)
    df.loc[df[outcome] > MAX_NNT] = MAX_NNT + 1
  sns.violinplot(data=df, x=event_col+'_HOUR', y=outcome, ax=ax)
  ax.set_title(f'{event} Hour of Day & {outcome} Distribution', fontsize=19)
  ax.set_xlabel('Hour of Day', fontsize=16)
  ax.set_ylabel(outcome, fontsize=16)
  plt.show()


# ---------------------------------- Patient hospital event & time of day ----------------------------------
# histograms (admit, discharge, end of surgery, wheel-out time ...)
def time_of_day_historgram(df, event='ADMIT'):
  total = df.SURG_CASE_KEY.nunique()
  alpha = 0.9
  if event == ADMIT_DTM:
    event_df = df.groupby(df[ADMIT_DTM].dt.hour).SURG_CASE_KEY.count()
    event_txt = "Admission"
    alpha = 0.65
  elif event == WHEELOUT:
    event_df = df.groupby(df.ACTUAL_STOP_DATE_TIME.dt.hour).SURG_CASE_KEY.count()
    event_txt = "Wheel-out"
  elif event == DISCHARGE_DTM:
    event_df = df.groupby(df[DISCHARGE_DTM].dt.hour).SURG_CASE_KEY.count()
    event_txt = "Discharge"
  elif event == IPSTART:
    event_df = df.groupby(df.INCISION_PROCEDURE_START_DATE_TIME.dt.hour).SURG_CASE_KEY.count()
    event_txt = "Incision Procedure Start"
  elif event == SURG_START_DTM:
    event_df = df.groupby(df[SURG_START_DTM].dt.hour).SURG_CASE_KEY.count()
    event_txt = "Surgeon Start"
  elif event == SURG_END_DTM:
    event_df = df.groupby(df[SURG_END_DTM].dt.hour).SURG_CASE_KEY.count()
    event_txt = "Surgery End"
  else:
    raise KeyError("%s is not supported for EDA" % event)

  for h in range(24):
    if event_df.get(h) is None:
      event_df[h] = 0

  event_cnts = [event_df[i] for i in range(24)]
  fig, ax = plt.subplots(1, 1, figsize=(12, 8))
  ax.bar(np.arange(24)-0.5, event_cnts, alpha=alpha, width=0.76)
  if event == DISCHARGE_DTM:
    ax.axvline(x=11, ls='--', lw=1.5, color='red', alpha=0.8, label='Time of Discharge Notice')
  ax.set_xticks(np.arange(24))
  ax.set_title("%s Time Histogram" % event_txt, fontsize=19, y=1.01)
  ax.set_xlabel("%s hour of day" % event_txt, fontsize=16)
  ax.set_ylabel("Number of cases", fontsize=16)
  ax.yaxis.set_tick_params(labelsize=13)
  ax.xaxis.set_tick_params(labelsize=13)
  ax.legend(prop={'size': 14})

  i = 0
  rects = ax.patches
  for rect in rects:
    if event_df[i] > 0:
      ht, wd = rect.get_height(), rect.get_width()
      x, y = rect.get_xy()
      ax.text(x + wd / 2, ht + 20, "{:.1%}".format(event_df[i] / total),
              ha='center', va='bottom', fontsize=8)
    i += 1
  plt.savefig(f'{event}.png', dpi=500)
  plt.show()


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
    # TODO: Figure out how to handle edge cases!
    #post_los_df = post_los_df[df.DISCHARGE_DATE_TIME.dt.hour]
    unit_txt = "night"
    pass
  else:
    raise ValueError("Unit %d is not supported!" % counting_unit)
  return  post_los_df, unit_txt


def postop_los_histogram(df, unit="H", exclude_outliers=0, plot_los=False):
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
    ax.hist(post_los_df, bins, color="orange", edgecolor='black', alpha=0.9, linewidth=0.5)

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
def op_length_postopLos_eda(dashb_df):
  df = dashb_df[[LOS, SURG_START_DTM, SURG_END_DTM, DISCHARGE_DTM]]
  df['op_length'] = pd.to_datetime(dashb_df[SURG_END_DTM]) - pd.to_datetime(dashb_df[SURG_START_DTM]) / np.timedelta64(1, 'H')
  postop_los_df, unit_txt = gen_postop_los(df)




# -------------------------------- Patient 30-day Revisit Flag - LOS distribution --------------------------------
def gen_revisit_col(dashb_df):
  # MRN, SURG_CASE_KEY, DISCHARGE_DTM, ADMIT_DTM
  df = dashb_df.loc[:, ['MRN', 'SURG_CASE_KEY', ADMIT_DTM, DISCHARGE_DTM]]
  df = df.join(df.groupby('MRN').count()['SURG_CASE_KEY'].reset_index(name='visit_cnt').set_index('MRN'), on='MRN', how='left')
  df = df[df['visit_cnt'] > 1]
  surg_key_to_revisit30 = defaultdict(int)
  for mrn in df['MRN'].unique():
    mrn_df = df.loc[df['MRN'] == mrn, ['SURG_CASE_KEY', ADMIT_DTM, DISCHARGE_DTM]].sort_values(by=ADMIT_DTM)
    prev_discharge_dtm = pd.to_datetime(mrn_df.iloc[0][DISCHARGE_DTM])
    for i in range(1, len(mrn_df)):
      admit_dtm = pd.to_datetime(mrn_df.iloc[i][ADMIT_DTM])
      if (admit_dtm - prev_discharge_dtm) / np.timedelta64(1, 'D') <= 30:
        print(admit_dtm, prev_discharge_dtm)
        surg_key_to_revisit30[mrn_df.iloc[i]['SURG_CASE_KEY']] = 1
      prev_discharge_dtm = pd.to_datetime(mrn_df.iloc[i][DISCHARGE_DTM])

  return surg_key_to_revisit30



def revisit30_eda(dashb_df, outcome=NNT, preprocess_y=True):
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
  ax.bar(xs-0.2, [100 * revisit_counter[x] / revisit_total for x in xs], label='30-day Readmission', width=0.4, alpha=0.8,
         color='salmon')
  ax.bar(xs+0.2, [100 * no_revisit_counter[x] / non_revisit_total for x in xs], label='Non-readmission', width=0.4, alpha=0.8,
         color='steelblue')
  ax.set_title('LOS Distribution by Readmission Indicator', fontsize=19, y=1.01)
  ax.set_xlabel('Length of stay', fontsize=16)
  ax.set_ylabel('Case count percentage (%)', fontsize=16)
  ax.yaxis.set_tick_params(labelsize=14)
  ax.legend(prop={'size': 13})

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
