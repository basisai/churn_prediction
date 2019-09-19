"""
Script to preprocess subscribers.
"""
import os
from pyspark.sql import functions as F

from .constants import AREA_CODES, STATES, FEATURE_COLS, TARGET_COL

BIGQUERY_PROJECT = os.getenv("BIGQUERY_PROJECT")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET")
RAW_SUBSCRIBER_TABLE = os.getenv("RAW_SUBSCRIBER_TABLE")
RAW_ALL_CALLS_TABLE = os.getenv("RAW_ALL_CALLS_TABLE")

BQ_TABLE = "{project}.{dataset}.{table}"


def preprocess_subscriber(spark):
    """Preprocess subscriber data."""
    # Load subscribers
    subscriber_table = BQ_TABLE.format(
        project=BIGQUERY_PROJECT,
        dataset=BIGQUERY_DATASET,
        table=RAW_SUBSCRIBER_TABLE,
    )
    subscribers_df = (
        spark.read.format("bigquery").option("table", subscriber_table).load()
        .withColumn("Intl_Plan", F.when(F.col("Intl_Plan") == "yes", 1).otherwise(0))
        .withColumn("VMail_Plan", F.when(F.col("VMail_Plan") == "yes", 1).otherwise(0))
        .withColumn("Churn", F.when(F.col("Churn") == "yes", 1).otherwise(0))
    )

    # Load raw calls
    all_calls_table = BQ_TABLE.format(
        project=BIGQUERY_PROJECT,
        dataset=BIGQUERY_DATASET,
        table=RAW_ALL_CALLS_TABLE,
    )
    calls_df = (
        spark.read.format("bigquery").option("table", all_calls_table).load()
        .groupBy("User_id")
        .pivot("Call_type", ["Day", "Eve", "Night", "Intl"])
        .agg(F.sum("Duration").alias("Mins"), F.count("Duration").alias("Calls"))
    )

    # Join subscribers with calls
    joined_df = subscribers_df.join(calls_df, on="User_id", how="left")
    joined_df = joined_df.fillna(0)
    return joined_df


def generate_features(spark):
    """Generate features."""
    joined_df = preprocess_subscriber(spark)
    for area_code in AREA_CODES:
        joined_df = joined_df.withColumn(
            "Area_Code_{}".format(area_code),
            F.when(F.col("Area_Code") == area_code, 1).otherwise(0)
        )

    for state in STATES:
        joined_df = joined_df.withColumn(
            "State_{}".format(state),
            F.when(F.col("State") == state, 1).otherwise(0)
        )

    joined_df = joined_df.select(FEATURE_COLS + [TARGET_COL])
    return joined_df
