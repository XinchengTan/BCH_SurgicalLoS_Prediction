
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import hvplot
# import panel as pn

from collections import Counter, defaultdict
from globals import *
from c1_data_preprocessing import gen_y_nnt



def cpt_count_eda(dashb_df, outcome=NNT, preprocess_y=True):
  figs, axs = plt.subplots(2, 1, figsize=(16, 18))
  df = dashb_df[[outcome, CPTS]]
  df['cpt_cnt'] = df[CPTS].apply(lambda x: len(set(x)))
  if preprocess_y and outcome == NNT:
    df[outcome] = gen_y_nnt(df[outcome])
    axs[1].set_yticks(sorted(df[outcome].unique()))
    axs[1].set_yticklabels(NNT_CLASS_LABELS, fontsize=14)

  cap = 20
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
    axs[0].text(rect.get_x() + wd / 2, ht + 2.5, label,
                ha='center', va='bottom', fontsize=9)

  # cap = 13
  df['capped_cnt1'] = df['cpt_cnt'].apply(lambda x: x if x < cap else cap)
  sns.violinplot(data=df, x='capped_cnt1', y=outcome)
  axs[1].set_title('LOS Distribution over CPT Count', fontsize=18, y=1.01)
  axs[1].set_xlabel(f'CPT count per case', fontsize=16)
  axs[1].set_ylabel('Length of stay', fontsize=16)
  axs[1].yaxis.set_tick_params(labelsize=14)
  axs[1].set_xticklabels(list(map(int, sorted(df['capped_cnt1'].unique())[:-1])) + [r'$\geq$%d' % cap],
                         fontsize=14)



# def display_pproc_profile(df: pd.DataFrame, freq_k=30, outcome_type=globals.LOS):
#   """
#   Profile df by visualizing each primary proc LoS distribution
#
#   :param df: a preprocessed dataframe
#   :param freq_k: number of most frequent primary procedure groups to visualize
#   :return:
#   """
#   pproc_df = df[['PRIMARY_PROC', outcome_type]].groupby(by=['PRIMARY_PROC'])
#
#   pproc_slider = pn.widgets.DiscreteSlider(name='Primary procedure', options=[], value='')
#
#   return


