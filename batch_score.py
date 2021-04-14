"""
Script to perform batch scoring.
"""
import os
import pickle
import time

import pandas as pd

from utils.constants import FEATURE_COLS

OUTPUT_MODEL_NAME = os.getenv("OUTPUT_MODEL_NAME")
FEATURES_DATA = os.path.join(
    os.getenv("TEMP_DATA_BUCKET"), os.getenv("FEATURES_DATA"))
SUBSCRIBER_SCORE_DATA = os.path.join(
    os.getenv("TEMP_DATA_BUCKET"), os.getenv("SUBSCRIBER_SCORE_DATA"))


def main():
    """Batch scoring pipeline"""
    print("\tLoading active subscribers")
    subscriber_df = pd.read_csv(FEATURES_DATA)
    subscriber_pd_df = (
        subscriber_df
        .query("Churn==0")
        .drop(columns=["Churn"])
    )
    print(f"\tNumber of active subscribers = {len(subscriber_pd_df)}")

    print("\tLoading model")
    with open("/artefact/" + OUTPUT_MODEL_NAME, "rb") as model_file:
        clf = pickle.load(model_file)

    print("\tScoring")
    subscriber_pd_df["Prob"] = (
        clf.predict_proba(subscriber_pd_df[FEATURE_COLS])[:, 1]
    )

    print("\tSaving scores")
    subscriber_pd_df[["User_id", "Prob"]].to_csv(
        SUBSCRIBER_SCORE_DATA,
        index=False,
    )


if __name__ == "__main__":
    main()
