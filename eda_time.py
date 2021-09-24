"""
EDA on date-time related features,
such as time of admission, discharge, end of surgery, actual start (wheel-in), actual stop (wheel out) etc.
"""
import matplotlib.pyplot as plt
import numpy as np

from . import globals


# ---------------------------------- Patient hospital event time of day ----------------------------------
# histograms (admit, discharge, end of surgery, wheel-out time ...)
def time_of_day_historgram(df, event='ADMIT'):
  total = df.SURG_CASE_KEY.nunique()
  if event == globals.WHEELIN or event == globals.ADMIT:
    event_df = df.groupby(df.ACTUAL_START_DATE_TIME.dt.hour).SURG_CASE_KEY.count()
    event_txt = "Wheel-in"
  elif event == globals.WHEELOUT:
    event_df = df.groupby(df.ACTUAL_STOP_DATE_TIME.dt.hour).SURG_CASE_KEY.count()
    event_txt = "Wheel-out"
  elif event == globals.DISCHARGE:
    event_df = df.groupby(df.DISCHARGE_DATE_TIME.dt.hour).SURG_CASE_KEY.count()
    event_txt = "Discharge"
  elif event == globals.IPSTART:
    event_df = df.groupby(df.INCISION_PROCEDURE_START_DATE_TIME.dt.hour).SURG_CASE_KEY.count()
    event_txt = "Incision Procedure Start"
  elif event == globals.STARTOS:
    event_df = df.groupby(df.SURGEON_START_DATE_TIME.dt.hour).SURG_CASE_KEY.count()
    event_txt = "Surgeon Start"
  elif event == globals.ENDOS:
    event_df = df.groupby(df.SURGERY_END_DATE_TIME.dt.hour).SURG_CASE_KEY.count()
    event_txt = "Surgery End"
  else:
    raise KeyError("%s is not supported for EDA" % event)

  for h in range(24):
    if event_df.get(h) is None:
      event_df[h] = 0

  event_cnts = [event_df[i] for i in range(24)]
  plt.rcParams['figure.figsize'] = (11, 7)
  graph = plt.bar(range(24), event_cnts)
  plt.xticks(np.arange(24), labels=np.arange(24))
  plt.title("%s Time - Cases Histogram" % event_txt, fontsize=15)
  plt.xlabel("%s Hour" % event_txt, fontsize=13)
  plt.ylabel("Number of Surgical Cases", fontsize=13)

  i = 0
  for p in graph:
    if event_df[i] > 0:
      width = p.get_width()
      height = p.get_height()
      x, y = p.get_xy()

      plt.text(x + width / 2,
               y + height * 1.02,
               "{:.2%}".format(event_df[i] / total),
               ha='center', fontsize=8)
    i += 1

  plt.show()


# ---------------------------------- Postoperative LoS Histogram ----------------------------------
# Calculate postoperative LoS
def gen_postop_los(df, counting_unit="H"):
  post_los_df = (df.DISCHARGE_DATE_TIME - df.ACTUAL_STOP_DATE_TIME).astype('timedelta64[ns]')
  # print("before rounding", post_los_df.head(6))
  if counting_unit == globals.HOUR:
    post_los_df = post_los_df.dt.round('1h').astype('timedelta64[h]')
    unit_txt = "hour"
    # print("after rounding: ", post_los_df.head(6))
  elif counting_unit == globals.DAY:
    post_los_df = post_los_df.dt.round('1h').astype('timedelta64[D]')
    unit_txt = "day"
  elif counting_unit == globals.NIGHT:
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
    los_df = df.LENGTH_OF_STAY if unit == globals.DAY else df.LENGTH_OF_STAY * 24
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


def los_postop_los_scatter(df, unit=globals.HOUR, xylim=None):
  postop_los_df, unit_txt = gen_postop_los(df, counting_unit=unit)
  los_df = df.LENGTH_OF_STAY * 24 if unit == globals.HOUR else df.LENGTH_OF_STAY

  fig, ax = plt.subplots(figsize=(11,10))
  ax.scatter(los_df, postop_los_df, s=20, facecolors='none', edgecolors='purple')
  ax.set_title("LoS VS Postoperative LoS (%s)" % unit_txt, fontsize=16)
  ax.set_xlabel("LoS (%s)" % unit_txt, fontsize=14)
  ax.set_ylabel("Postoperative LoS (%s)" % unit_txt, fontsize=14)
  if xylim:
    ax.set_xlim([0, xylim])
    ax.set_ylim([0, xylim])
  lims = np.array([np.min([ax.get_xlim(), ax.get_ylim()]),
                   np.max([ax.get_xlim(), ax.get_ylim()])])
  ax.plot(lims, lims, '--', color='k', linewidth=3, alpha=0.8)
  if unit == globals.HOUR:
    ax.plot(lims, lims - 10, '--', color='r', linewidth=2.5, alpha=0.8)

  plt.show()


# def los_histogram(df, unit="D", exclude_outliers=0):
#
#
#   minLos, maxLos = min(df.LENGTH_OF_STAY), max(df.LENGTH_OF_STAY)
#   if exclude_outliers > 0:
#
#     maxLos = max(los_df)
#
#   fig, ax = plt.subplots(figsize=(12, 7))
#   bins = np.linspace(minLos, maxLos + globals.DELTA, 100)
#   ax.set_xlim([bins[0], bins[-1]])
#   ax.hist(df.LENGTH_OF_STAY, bins, color="red", alpha=0.65, edgecolor='black', linewidth=0.5)
#   plt.title("LoS Histogram")
#   plt.xlabel("LoS (%s)" % unit_txt)
#   plt.ylabel("Number of surgical cases")
#   plt.show()
#
#   return