"""
Script to train model.
"""
import logging
import os
import pickle

from bedrock_client.bedrock.api import BedrockApi
import numpy as np
import pandas as pd
import lightgbm as lgb
from prometheus_client import Histogram, generate_latest
from sklearn import metrics
from sklearn.model_selection import train_test_split

from utils.helper import get_temp_data_bucket
from utils.constants import FEATURE_COLS, SUBSCRIBER_FEATURES, TARGET_COL

TEMP_DATA_BUCKET = get_temp_data_bucket()
FEATURES_DATA = TEMP_DATA_BUCKET + os.getenv("FEATURES_DATA")
LR = float(os.getenv("LR"))
NUM_LEAVES = int(os.getenv("NUM_LEAVES"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS"))
OUTPUT_MODEL_NAME = os.getenv("OUTPUT_MODEL_NAME")


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
    print("Accuracy = {:.6f}".format(acc))
    print("Precision = {:.6f}".format(precision))
    print("Recall = {:.6f}".format(recall))
    print("F1 score = {:.6f}".format(f1_score))
    print("AUC = {:.6f}".format(auc))
    print("Average precision = {:.6f}".format(avg_prc))

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("Precision", precision)
    bedrock.log_metric("Recall", recall)
    bedrock.log_metric("F1 score", f1_score)
    bedrock.log_metric("AUC", auc)
    bedrock.log_metric("Avg precision", avg_prc)
    bedrock.log_chart_data(y_val.astype(int).tolist(),
                           y_prob.flatten().tolist())


def main():
    """Train pipeline"""
    model_data = pd.read_csv(FEATURES_DATA)

    print("\tSplitting train and validation data")
    x_train, x_val, y_train, y_val = train_test_split(
        model_data[FEATURE_COLS],
        model_data[TARGET_COL],
        test_size=0.2,
    )
    print("\tTrain model")
    gbm = lgb.LGBMClassifier(
        num_leaves=NUM_LEAVES,
        learning_rate=LR,
        n_estimators=N_ESTIMATORS,
    )
    gbm.fit(x_train, y_train)

    for i, k in enumerate(SUBSCRIBER_FEATURES):
        bins, width = _freedman_diaconis_bins(model_data[k])
        print(f"bins: {bins} width: {width}")
        bins = min(bins, 50)
        first = max(model_data[k].mean() - width * bins / 2, width)
        last = first + width * bins
        print(f"first: {first} last: {last}")
        metric = Histogram(
            name=f"feature_{i}_value",
            documentation=f"Real time values for feature index: {i}",
            # buckets=tuple(float(b) for b in os.getenv(
            #     "FEATURE_BINS", "0,0.25,0.5,0.75,1,2,5,10").split(",")),
            buckets=tuple([0] + list(np.linspace(start=first, stop=last, num=bins)))
        )
        for v in model_data[k]:
            metric.observe(v)
    y_prob = gbm.predict_proba(x_val)[:, 1]
    metric = Histogram(
        name=f"inference_value_test",
        documentation=f"Inference value of the test set",
        buckets=tuple(i / 10 for i in range(10))
    )
    for v in y_prob:
        metric.observe(v)
    # push_to_gateway(gateway="prometheus-pushgateway.core.svc", job="run_step_id", registry=REGISTRY)
    print(generate_latest())

    compute_log_metrics(gbm, x_val, y_val)

    print("\tSaving model")
    os.mkdir("/artefact/train")
    with open("/artefact/train/" + OUTPUT_MODEL_NAME, "wb") as model_file:
        pickle.dump(gbm, model_file)


def _freedman_diaconis_bins(a):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    # From https://stats.stackexchange.com/questions/798/
    if len(a) < 2:
        return 1
    iqr = a.quantile(q=0.75) - a.quantile(q=0.25)
    h = 2 * iqr / (len(a) ** (1 / 3))
    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        bins = int(np.sqrt(a.size))
        return bins, (a.max() - a.min()) / bins
    else:
        return int(np.ceil((a.max() - a.min()) / h)), h

if __name__ == "__main__":
    main()
