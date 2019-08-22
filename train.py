"""
Script to train model.
"""
import logging
import os
import pickle

import lightgbm as lgb
import requests
from bedrock_client.bedrock.api import BedrockApi
from pyspark.sql import SparkSession
from sklearn import metrics
from sklearn.model_selection import train_test_split

from utils.constants import FEATURE_COLS
from utils.preprocess import generate_features

LR = float(os.getenv("LR"))
NUM_LEAVES = int(os.getenv("NUM_LEAVES"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS"))
OUTPUT_MODEL_NAME = os.getenv("OUTPUT_MODEL_NAME")
TARGET_COL = "Churn"


def download_artefact(pipeline_run_id: str):
    print(f"Downloading artefact for pipeline run: {pipeline_run_id}")
    filename = f"/tmp/{pipeline_run_id}-artefact.zip"
    with requests.get(
        f"https://api.amoy.ai/v1/artefact/{pipeline_run_id}/internal",
        headers={"X-Bedrock-Api-Token": os.environ["BEDROCK_API_TOKEN"]},
        stream=True,
    ) as response:
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    return filename


def download_artefact_from_latest_run(pipeline_public_id: str):
    response = requests.get(
        f"https://api.amoy.ai/v1/training_pipeline/{pipeline_public_id}/run/",
        headers={"X-Bedrock-Access-Token": os.environ["BEDROCK_ACCESS_TOKEN"]},
        timeout=30,
    )
    response.raise_for_status()
    runs = response.json()
    if not runs:
        print(f"No runs to download: {pipeline_public_id}")
        return
    last_run = max(runs, key=lambda run: run["created_at"])
    filename = download_artefact(last_run["entity_id"])
    print(f"Downloaded: {filename}")


def compute_log_metrics(gbm, x_val, y_val):
    """Compute and log metrics."""
    print("\tEvaluating using validation data")
    y_prob = gbm.predict_proba(x_val)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    acc = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    auc = metrics.roc_auc_score(y_val, y_prob)
    avg_prc = metrics.average_precision_score(y_val, y_prob)
    print("Accuracy =", acc)
    print("Precision =", precision)
    print("Recall =", recall)
    print("F1 score =", f1_score)
    print("AUC =", auc)
    print("Average precision =", avg_prc)

    # Log metrics
    logger = logging.getLogger(__name__)

    bedrock = BedrockApi(logger)
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("Precision", precision)
    bedrock.log_metric("Recall", recall)
    bedrock.log_metric("F1 score", f1_score)
    bedrock.log_metric("AUC", auc)
    bedrock.log_metric("Avg precision", avg_prc)
    bedrock.log_chart_data(y_val.astype(int).tolist(), y_prob.flatten().tolist())


def main():
    """Train pipeline"""
    print("\tGenerating features")
    with SparkSession.builder.appName("Preprocessing").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("FATAL")
        model_data = generate_features(spark).drop("User_id").toPandas()

    print("\tSplitting train and validation data")
    x_train, x_val, y_train, y_val = train_test_split(
        model_data[FEATURE_COLS].values,
        model_data[TARGET_COL].values,
        test_size=0.2,
        random_state=42,
    )

    print("\tTrain model")
    gbm = lgb.LGBMClassifier(
        num_leaves=NUM_LEAVES, learning_rate=LR, n_estimators=N_ESTIMATORS
    )
    gbm.fit(x_train, y_train)
    compute_log_metrics(gbm, x_val, y_val)

    print("\tSaving model")
    with open("/artefact/" + OUTPUT_MODEL_NAME, "wb") as model_file:
        pickle.dump(gbm, model_file)


if __name__ == "__main__":
    main()
