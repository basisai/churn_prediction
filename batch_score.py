"""
Script to perform batch scoring.
"""
import os
import pickle
import time

import pandas as pd

from utils.constants import FEATURE_COLS

FEATURES_DATA = os.path.join(os.getenv("TEMP_DATA_BUCKET"), os.getenv("FEATURES_DATA"))
OUTPUT_MODEL_NAME = os.getenv("OUTPUT_MODEL_NAME")
BIGQUERY_PROJECT = os.getenv("BIGQUERY_PROJECT")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET")
DEST_SUBSCRIBER_SCORE_TABLE = os.getenv("DEST_SUBSCRIBER_SCORE_TABLE")


def main():
    """Batch scoring pipeline"""
    start = time.time()
    print("\tLoading active subscribers")
    subscriber_df = pd.read_csv(FEATURES_DATA)
    subscriber_pd_df = (
        subscriber_df
        .query("Churn==0")
        .drop(columns=["Churn"])
    )
    print("\tTime taken = {:.2f} min".format((time.time() - start) / 60))
    print("\tNumber of active subscribers = {}"
            .format(len(subscriber_pd_df)))

    print("\tLoading model")
    with open("/artefact/train/" + OUTPUT_MODEL_NAME, "rb") as model_file:
        gbm = pickle.load(model_file)

    print("\tScoring")
    subscriber_pd_df["Prob"] = (
        gbm.predict_proba(subscriber_pd_df[FEATURE_COLS])[:, 1]
    )

    start = time.time()
    print("\tSaving scores to BigQuery")
    subscriber_pd_df[["User_id", "Prob"]].to_gbq(
        f"{BIGQUERY_DATASET}.{DEST_SUBSCRIBER_SCORE_TABLE}",
        project_id=BIGQUERY_PROJECT,
        if_exists="replace",
    )
    print("\tTime taken = {:.2f} min".format((time.time() - start) / 60))


if __name__ == "__main__":
    main()
