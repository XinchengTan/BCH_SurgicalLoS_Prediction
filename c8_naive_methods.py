import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from typing import List

from c1_data_preprocessing import Dataset
import globals
from c4_model_perf import MyScorer
import jyp4_model_eval as model_eval


def gen_pproc_decile(train_df, decile_outcome=globals.LOS):
  # Calculate median LoS of each pproc
  pproc_median = train_df[['PRIMARY_PROC', decile_outcome]].groupby(by=['PRIMARY_PROC'])[decile_outcome].median()

  # Round each median to int and clip it between 0 and MAX_NNT + 1
  pproc_sev_score = pd.DataFrame(pproc_median.apply(lambda x: float(min(round(x), globals.MAX_NNT + 1))))\
    .rename(columns={decile_outcome: 'PPROC_DECILE'})
  pproc_sev_score.reset_index(inplace=True)

  return pproc_sev_score


def predict_with_pproc_decile(test_df, pproc_sev_score_df):
  new_test_df = test_df.join(pproc_sev_score_df.set_index('PRIMARY_PROC'), on='PRIMARY_PROC', how='inner')

  return new_test_df #['PPROC_DECILE'].to_list()


def predict_with_pproc_decile_cv(datasets: List[Dataset], decile_outcome, pred_outcome=globals.NNT):
  preds = []
  y_true = []
  for dataset in datasets:
    pproc_sev_score_df = gen_pproc_decile(dataset.train_cohort_df, decile_outcome)
    new_test_df = predict_with_pproc_decile(dataset.test_cohort_df, pproc_sev_score_df)
    y, pred = new_test_df[pred_outcome], new_test_df['PPROC_DECILE']
    preds.extend(pred.to_list())
    y_true.extend(y.to_list())

  # Basic evaluation
  preds, y_true = np.array(preds), np.array(y_true)
  print("Accuracy: %.2f%%" % (100 * accuracy_score(y_true, preds)))
  print("Accuracy (tol=1 night): %.2f%%" % (100 * MyScorer.scorer_1nnt_tol(y_true, preds)))

  _ = model_eval.gen_confusion_matrix(y_true, preds, 'PPROC Decile', Xtype='Full dataset (%d-fold)' % len(datasets))
  return preds


def predict_with_pproc_mode():

  return


def predict_with_voting_cpt_decile():

  return
