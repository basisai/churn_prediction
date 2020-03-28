"""
Script for serving.
"""
import os
import pickle

import numpy as np
from flask import Flask, request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Histogram,
    generate_latest
)
from prometheus_client.multiprocess import MultiProcessCollector

from metrics import HISTOGRAM_PATH, serve_histogram
from utils.constants import AREA_CODES, STATES, SUBSCRIBER_FEATURES

OUTPUT_MODEL_NAME = "/artefact/train/lgb_model.pkl"

REGISTRY = CollectorRegistry()
MultiProcessCollector(REGISTRY)
FEATURE_HISTOGRAM = serve_histogram(registry=REGISTRY)
INFERENCE_HISTOGRAM = Histogram(
    name=f"inference_value",
    documentation=f"Real time inference value from model server",
    buckets=tuple(i / 10 for i in range(10)),
)
with open(HISTOGRAM_PATH, "rb") as f:
    BASELINE_METRICS = f.read()


def predict_prob(subscriber_features,
                 model=pickle.load(open(OUTPUT_MODEL_NAME, "rb"))):
    """Predict churn probability given subscriber_features.

    Args:
        subscriber_features (dict)
        model

    Returns:
        churn_prob (float): churn probability
    """
    row_feats = list()
    for col in SUBSCRIBER_FEATURES:
        row_feats.append(subscriber_features[col])

    for area_code in AREA_CODES:
        if subscriber_features["Area_Code"] == area_code:
            row_feats.append(1)
        else:
            row_feats.append(0)

    for state in STATES:
        if subscriber_features["State"] == state:
            row_feats.append(1)
        else:
            row_feats.append(0)

    for j, histogram in enumerate(FEATURE_HISTOGRAM):
        if j >= len(row_feats):
            break
        histogram.observe(row_feats[j])

    # Score
    churn_prob = (
        model
        .predict_proba(np.array(row_feats).reshape(1, -1))[:, 1]
        .item()
    )

    INFERENCE_HISTOGRAM.observe(churn_prob)
    return churn_prob


# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
def get_churn():
    """Returns the `churn_prob` given the subscriber features"""

    subscriber_features = request.json
    result = {
        "churn_prob": predict_prob(subscriber_features)
    }
    return result


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Returns real time feature values recorded by prometheus
    """
    data = BASELINE_METRICS + generate_latest(REGISTRY)
    resp = Response(data)
    resp.headers['Content-Type'] = CONTENT_TYPE_LATEST
    resp.headers['Content-Length'] = str(len(data))
    return resp


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()
