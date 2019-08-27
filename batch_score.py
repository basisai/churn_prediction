"""
Script to perform batch scoring.
"""
import os
import pickle
import time

from pyspark.sql import SparkSession

from utils.constants import FEATURE_COLS
from utils.preprocess import generate_features

OUTPUT_MODEL_NAME = os.getenv("OUTPUT_MODEL_NAME")
DEST_BIGQUERY_PROJECT = os.getenv("RAW_BIGQUERY_PROJECT")
DEST_BIGQUERY_DATASET = os.getenv("RAW_BIGQUERY_DATASET")
DEST_SUBSCRIBER_SCORE_TABLE = os.getenv("DEST_SUBSCRIBER_SCORE_TABLE")


def main():
    """Batch scoring pipeline"""
    with SparkSession.builder.appName("BatchScoring").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("FATAL")

        start = time.time()
        print("\tLoading active subscribers")
        subscriber_df = generate_features(spark)
        subscriber_pd_df = (
            subscriber_df
            .filter(subscriber_df["Churn"] == 0)
            .drop("Churn")
            .toPandas()
        )
        print("\tTime taken = {:.2f} min".format((time.time() - start) / 60))
        print("\tNumber of active subscribers = {}"
              .format(subscriber_pd_df.shape[0]))

    print("\tLoading model")
    with open("/artefact/" + OUTPUT_MODEL_NAME, "rb") as model_file:
        gbm = pickle.load(model_file)

    print("\tScoring")
    subscriber_pd_df["Prob"] = (
        gbm.predict_proba(subscriber_pd_df[FEATURE_COLS])[:, 1]
    )

    start = time.time()
    print("\tSaving scores to BigQuery")
    subscriber_pd_df[["User_id", "Prob"]].to_gbq(
        f"{DEST_BIGQUERY_DATASET}.{DEST_SUBSCRIBER_SCORE_TABLE}",
        project_id=DEST_BIGQUERY_PROJECT,
        if_exists="replace",
    )
    print("\tTime taken = {:.2f} min".format((time.time() - start) / 60))


if __name__ == "__main__":
    main()
