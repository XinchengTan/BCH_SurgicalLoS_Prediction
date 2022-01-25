"""
Example script to execute the LoS prediction pipeline
"""
from sklearn.metrics import accuracy_score

from deployment import los_prediction_deployment_script as los_pred_pipeline
from deployment.los_prediction_global_vars import *


if __name__ == "__main__":
  # 1. Create a preprocessed dataframe from test cases csv
  # - can set verbose = False to not print the dataframe details through basic preprocessing
  test_data_df = los_pred_pipeline.get_combined_dataframe(
    data_fp=DEPLOY_DATA_DIR / 'nov-dec-4.csv',
    cpt_fp=DEPLOY_DATA_DIR / 'cpt3.csv',
    cpt_grp_fp=DEPLOY_DEP_FILES_DIR / 'cpt2group.csv',
    diag_fp=DEPLOY_DATA_DIR / 'ccsr3.csv',
    for_deployment_eval=True,
    verbose=True)  # for local testing purpose, set for_deployment_eval = True to get outcome vec

  # 2. Further preprocess the dataframe into a numerical input matrix, saving SURG_CASE_KEY and feature names
  X, feature_cols, X_case_keys = los_pred_pipeline.get_feature_matrix(
    test_data_df,
    skip_cases_fp=DEPLOY_DEP_FILES_DIR / 'skip_cases.csv',
    cols=FEATURE_COLS_NO_WEIGHT,
    onehot_cols=['PRIMARY_PROC', 'CPTS', 'CCSRS'], onehot_dtypes=[str, list, list],
    discretize_cols=['AGE_AT_PROC_YRS'])

  # 3. Load pretrained model
  # The following is an ensemble model. To use a single model, simply specify one key-value pair in 'md2model_filename'
  md2model_filename = {LGR: 'lgr.joblib',
                       KNN: 'knn.joblib',
                       DTCLF: 'dtclf.joblib',
                       RMFCLF: 'rmfclf.joblib',
                       GBCLF: 'gbclf.joblib'}
  model = los_pred_pipeline.load_pretrained_model(md2model_filename)

  # 4. Make prediction
  md_predictions = los_pred_pipeline.predict_los(X, model)

  # [Optional] 5. To evaluate the model performance on the past few months' dataset, create a outcome vector
  y_true = los_pred_pipeline.get_outcome_vec(test_data_df, X_case_keys)
  print("\nModel Accuracy: %.2f%%" % (accuracy_score(y_true, md_predictions) * 100))
  print("Model Accuracy (tol=1 night): %.2f%%" % (100 * los_pred_pipeline.scorer_1nnt_tol(y_true, md_predictions)))

  # [Optional] 6. Evaluate surgeon's prediction
  sur_pred = los_pred_pipeline.get_surgeon_pred_vec(test_data_df, X_case_keys)
  print("\nSurgeon Prediction Accuracy: %.2f%%" % (100 * accuracy_score(y_true, sur_pred)))
  print("Surgeon Prediction Accuracy (tol=1 night): %.2f%%" % (100 * los_pred_pipeline.scorer_1nnt_tol(y_true, sur_pred)))

