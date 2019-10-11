"""
Script for serving.
"""
import pickle

import numpy as np
from flask import Flask, request

from utils.constants import AREA_CODES, STATES, SUBSCRIBER_FEATURES

OUTPUT_MODEL_NAME = "/artefact/lgb_model.pkl"


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

    # Score
    churn_prob = (
        model
        .predict_proba(np.array(row_feats).reshape(1, -1))[:, 1]
        .item()
    )

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


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()
