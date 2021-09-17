"""
EDA on date-time related features,
such as time of admission, discharge, end of surgery, actual start (wheel-in), actual stop (wheel out) etc.
"""
import matplotlib.pyplot as plt
import numpy as np

from . import globals


# Patient hospital event time of day histogram (admit, discharge, end of surgery, wheel-out time ...)
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


def postOp_los_histogram(df):

  return