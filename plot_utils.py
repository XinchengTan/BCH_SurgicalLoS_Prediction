from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
from sklearn import metrics

from . import globals


def plot_learning_curve(gs, x_param_name, x_params, md, scorers):
  gs_cv_results = gs.cv_results_
  plt.figure(figsize=(13, 9), facecolor='white')
  plt.title("GridSearchCV for %s over '%s'" % (globals.clf2name[md], x_param_name), fontsize=18, y=1.01)
  plt.xlabel(x_param_name)
  plt.ylabel("Score")

  ax = plt.gca()
  ax.set_xlim(x_params[0], x_params[-1]*1.01)
  ax.set_ylim(0.2, 1.05)
  ax.set_facecolor('white')
  for pos in ['top', 'left', 'bottom', 'right']:
    ax.spines[pos].set_edgecolor('black')
  ax.grid(color='k', linestyle=':', linewidth=0.5)

  # Get the regular numpy array from the MaskedArray
  X_axis = np.array(gs_cv_results["param_%s" % x_param_name].data, dtype=float)

  colors = ["orange", 'r', "g", "k"]
  for scorer, color in zip(sorted(scorers), colors[:len(scorers)]):
    for sample, style, marker, annot_delta in (("train", "--", ",", 1.01), ("test", "-", "o", 0.99)):
      sample_score_mean = gs_cv_results["mean_%s_%s" % (sample, scorer)]
      sample_score_std = gs_cv_results["std_%s_%s" % (sample, scorer)]
      if np.isnan(X_axis).any():
        idx = np.where(np.isnan(X_axis))[0][0]
        ax.scatter(x=0.5, y=sample_score_mean[idx], marker=marker, c=color)
        ax.text(0.6, sample_score_mean[idx]*annot_delta, '{}_{} = {:.1%}'.format(sample, scorer, sample_score_mean[idx]))
      ax.fill_between(
        X_axis,
        sample_score_mean - sample_score_std,
        sample_score_mean + sample_score_std,
        alpha=0.1 if sample == "test" else 0,
        color=color,
      )
      ax.plot(
        X_axis,
        sample_score_mean,
        style,
        color=color,
        alpha=1 if sample == "test" else 0.7,
        label="%s (%s)" % (scorer, sample),
      )

    best_index = np.nonzero(gs_cv_results["rank_test_%s" % scorer] == 1)[0][0]
    best_score = gs_cv_results["mean_test_%s" % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot(
      [
        X_axis[best_index],
      ]
      * 2,
      [0, best_score],
      linestyle="-.",
      color=color,
      marker="x",
      markeredgewidth=3,
      ms=8,
    )

    # Annotate the best score for that scorer
    ax.annotate("{:.1%}".format(best_score), (X_axis[best_index], best_score + 0.005), fontsize=16)

    # Plot doctor's estimation
    # Acc: 69.4%, 1-NNT Acc: 90.8%

  plt.legend(loc="best")
  plt.show()


def plot_error_histogram(true_y, pred_y, md_name, Xtype=globals.XTRAIN, yType="LoS (days)", ax=None, groupby_outcome=False):
  error = np.array(pred_y) - np.array(true_y)
  bins = np.arange(-globals.MAX_NNT, globals.MAX_NNT+1, 1)

  if ax is None:
    plt.hist(error, bins=bins, align='left', color='grey')
    plt.title("Prediction Error Histogram (%s - %s)" % (md_name, Xtype), y=1.01, fontsize=18)
    plt.xlabel(yType, fontsize=16)
    plt.xticks(bins)
    plt.ylabel("Number of surgical cases", fontsize=16)
    plt.show()
  else:
    ax.set_facecolor("white")
    ax.grid(color='k', linestyle=':', linewidth=0.5)
    for pos in ['top', 'left', 'bottom', 'right']:
      ax.spines[pos].set_edgecolor('black')
    if not groupby_outcome:
      counts, bins, patches = ax.hist(error, bins=bins, align='left', color='gray')
      rects = ax.patches
      labels = ["{:.1%}".format(cnt / len(true_y)) for cnt in counts]
      for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label,
                ha='center', va='bottom', fontsize=11)
    else:
      errs = []
      for i in range(globals.MAX_NNT+2):
        err = np.array(pred_y)[true_y == i] - np.array(true_y)[true_y == i]
        errs.append(err)
      labels = ['True NNT = %d' % i for i in range(globals.MAX_NNT+1)] + ['True NNT = %d+' % globals.MAX_NNT]
      counts, bins, patches = ax.hist(errs, bins=bins, align='left', label=labels)
      ax.legend()

    ax.set_title("Prediction Error Histogram (%s - %s)" % (md_name, Xtype), y=1.01, fontsize=18)
    ax.set_xlabel(yType, fontsize=16)
    ax.set_xticks(bins)
    ax.set_ylabel("Number of surgical cases", fontsize=16)
  return error


def plot_error_hist_pct(true_y, pred_y, md_name, Xtype=globals.XTRAIN, yType="LoS (days)", ax=None):
  ax.set_facecolor("white")
  for pos in ['top', 'left', 'bottom', 'right']:
    ax.spines[pos].set_edgecolor('black')
  ax.grid(color='k', linestyle=':', linewidth=0.5)
  bins = np.arange(-globals.MAX_NNT, globals.MAX_NNT+1, 1)

  wd = 0.1
  center_i = (globals.MAX_NNT+1) // 2
  outcome_cntr = Counter(true_y)
  for i in range(globals.MAX_NNT + 2):
    err = np.array(pred_y)[true_y == i] - np.array(true_y)[true_y == i]
    cnter = Counter(err)
    label = 'True NNT = %d' % i if i < globals.MAX_NNT + 1 else 'True NNT = %d+' % globals.MAX_NNT
    bar_ys = [100 * cnter[j] / outcome_cntr[i] if outcome_cntr[i] != 0 else 0 for j in bins]
    ax.bar(bins+wd*(i - center_i), bar_ys, width=0.1, label=label)

  ax.set_title("Prediction Error Histogram (%s - %s)" % (md_name, Xtype), y=1.01, fontsize=18)
  ax.set_xlabel(yType, fontsize=16)
  ax.set_xticks(bins)
  ax.set_ylabel("Percentage of the True Class Size (%)", fontsize=16)
  ax.legend()


def plot_calibration_basics(ax, cutoff=None, Xtype=globals.XTRAIN):
  ax.plot([0, 1], [0, 1], linestyle='--', color='black')
  title = "Cutoff=%d: Calibration Curves (%s)" % (cutoff, Xtype) if cutoff is not None else "Calibration Curves (%s)" % Xtype
  ax.set_title(title, y=1.01, fontsize=18)
  ax.set_xlabel("Mean Predicted Probability", fontsize=15)
  ax.set_ylabel("Fraction of Positives", fontsize=15)
  ax.legend(prop=dict(size=14))


def plot_roc_basics(ax, cutoff=None, Xtype=globals.XTRAIN):
  ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
  title = "Cutoff=%d: ROC Curve Comparison (%s)" % (cutoff, Xtype) if cutoff is not None else "ROC Curves (%s)" % Xtype
  ax.set_title(title, y=1.01, fontsize=18)
  ax.set_xlabel("False Positive Rate", fontsize=15)
  ax.set_ylabel("True Positive Rate", fontsize=15)
  ax.legend(prop=dict(size=14))


def plot_roc_best_threshold(X, y, clf, ax):
  fpr, tpr, thresholds = metrics.roc_curve(y, clf.predict_proba(X)[:, 1])
  ix = np.argmax(tpr - fpr)
  ax.scatter(fpr[ix], tpr[ix], s=[120], marker='*', color='red')


def plot_connectivity_graph(A, cmap=plt.cm.tab20, threshold=0, threshold_pct=0):
  plt.figure(figsize=(12, 9))

  G = nx.convert_matrix.from_numpy_array(A)
  # edge_weights = [A[i][j] for i, j in G.edges]
  edge_weights = [A[i][j] if A[i][j] > threshold else 1 for i, j in G.edges]

  node_colors = list(range(A.shape[0]))
  pos = nx.spring_layout(G, weight='edges')
  # pos = nx.spectral_layout(G, weight='edges')

  sc = nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors, cmap=cmap, node_size=800)
  nx.draw_networkx_edges(G, pos=pos, width=[np.log10(e) for e in edge_weights],
                         edge_color=edge_weights, edge_cmap=plt.cm.Greys)  # YlOrRd
  nx.draw_networkx_labels(G, pos=pos, labels={i: i + 1 for i in G.nodes})

  patches = []
  for nd in G.nodes:
    plt.plot([0], [0], color=cmap(nd), label=globals.diaglabels[nd])
    patches.append(mpatches.Patch(color=cmap(nd), label='%d. %s' % (nd + 1, globals.diaglabels[nd])))
  plt.legend(handles=patches, bbox_to_anchor=(1, 1.01))
  plt.axis('off')


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
  """
  Create a heatmap from a numpy array and two lists of labels.

  Parameters
  ----------
  data
      A 2D numpy array of shape (N, M).
  row_labels
      A list or array of length N with the labels for the rows.
  col_labels
      A list or array of length M with the labels for the columns.
  ax
      A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
      not provided, use current axes or create a new one.  Optional.
  cbar_kw
      A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
  cbarlabel
      The label for the colorbar.  Optional.
  **kwargs
      All other arguments are forwarded to `imshow`.
  """

  if not ax:
    ax = plt.gca()

  # Plot the heatmap
  im = ax.imshow(data, **kwargs)

  # Create colorbar
  cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
  cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=15)

  # We want to show all ticks...
  ax.set_xticks(np.arange(data.shape[1]))
  ax.set_yticks(np.arange(data.shape[0]))
  # ... and label them with the respective list entries.
  ax.set_xticklabels(col_labels)
  ax.set_yticklabels(row_labels)

  # Let the horizontal axes labeling appear on top.
  ax.tick_params(top=True, bottom=False,
                 labeltop=True, labelbottom=False)

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
           rotation_mode="anchor")

  # Turn spines off and create white grid. #ax.spines[:].set_visible(False)

  ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
  ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
  ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
  ax.tick_params(which="minor", bottom=False, left=False)

  return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
  """
  A function to annotate a heatmap.

  Parameters
  ----------
  im
      The AxesImage to be labeled.
  data
      Data used to annotate.  If None, the image's data is used.  Optional.
  valfmt
      The format of the annotations inside the heatmap.  This should either
      use the string format method, e.g. "$ {x:.2f}", or be a
      `matplotlib.ticker.Formatter`.  Optional.
  textcolors
      A pair of colors.  The first is used for values below a threshold,
      the second for those above.  Optional.
  threshold
      Value in data units according to which the colors from textcolors are
      applied.  If None (the default) uses the middle of the colormap as
      separation.  Optional.
  **kwargs
      All other arguments are forwarded to each call to `text` used to create
      the text labels.
  """

  if not isinstance(data, (list, np.ndarray)):
    data = im.get_array()

  # Normalize the threshold to the images color range.
  if threshold is not None:
    threshold = im.norm(threshold)
  else:
    threshold = im.norm(data.max()) / 2.

  # Set default alignment to center, but allow it to be
  # overwritten by textkw.
  kw = dict(horizontalalignment="center",
            verticalalignment="center")
  kw.update(textkw)

  # Get the formatter in case a string is supplied
  if isinstance(valfmt, str):
    valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

  # Loop over the data and create a `Text` for each "pixel".
  # Change the text's color depending on the data.
  texts = []
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
      text = im.axes.text(j, i, "", **kw)
      texts.append(text)

  return texts