"""
Script to preprocess subscribers.
"""
import os
from constants import AREA_CODES, STATES
from pyspark.sql import functions as F

RAW_BIGQUERY_PROJECT = os.getenv("RAW_BIGQUERY_PROJECT")
RAW_BIGQUERY_DATASET = os.getenv("RAW_BIGQUERY_DATASET")
RAW_SUBSCRIBER_TABLE = os.getenv("RAW_SUBSCRIBER_TABLE")
RAW_DAY_CALL_TABLE = os.getenv("RAW_DAY_CALL_TABLE")
RAW_EVE_CALL_TABLE = os.getenv("RAW_EVE_CALL_TABLE")
RAW_INTL_CALL_TABLE = os.getenv("RAW_INTL_CALL_TABLE")
RAW_NIGHT_CALL_TABLE = os.getenv("RAW_NIGHT_CALL_TABLE")

BQ_TABLE = "{project}.{dataset}.{table}"


def preprocess_subscribers(spark):
    """Preprocess subscribers."""
    # Load subscribers
    subscriber_table = BQ_TABLE.format(
        project=RAW_BIGQUERY_PROJECT,
        dataset=RAW_BIGQUERY_DATASET,
        table=RAW_SUBSCRIBER_TABLE,
    )
    subscribers_df = spark.read.format("bigquery").option("table", subscriber_table).load()

    subscribers_df = (
        subscribers_df
            .withColumn("Intl_Plan", F.when(F.col("Intl_Plan") == "yes", 1).otherwise(0))
            .withColumn("VMail_Plan", F.when(F.col("VMail_Plan") == "yes", 1).otherwise(0))
            .withColumn("Churn", F.when(F.isnull("Date_Closed"), 0).otherwise(1))
            .drop("Date_Created")
            .drop("Date_Closed")
            .drop("Phone")
    )

    # Load raw calls
    day_call_table = BQ_TABLE.format(
        project=RAW_BIGQUERY_PROJECT,
        dataset=RAW_BIGQUERY_DATASET,
        table=RAW_DAY_CALL_TABLE,
    )
    raw_day_calls_df = spark.read.format("bigquery").option("table", day_call_table).load()
    eve_call_table = BQ_TABLE.format(
        project=RAW_BIGQUERY_PROJECT,
        dataset=RAW_BIGQUERY_DATASET,
        table=RAW_EVE_CALL_TABLE,
    )
    raw_eve_calls_df = spark.read.format("bigquery").option("table", eve_call_table).load()
    intl_call_table = BQ_TABLE.format(
        project=RAW_BIGQUERY_PROJECT,
        dataset=RAW_BIGQUERY_DATASET,
        table=RAW_INTL_CALL_TABLE,
    )
    raw_intl_calls_df = spark.read.format("bigquery").option("table", intl_call_table).load()
    night_call_table = BQ_TABLE.format(
        project=RAW_BIGQUERY_PROJECT,
        dataset=RAW_BIGQUERY_DATASET,
        table=RAW_NIGHT_CALL_TABLE,
    )
    raw_night_calls_df = spark.read.format("bigquery").option("table", night_call_table).load()

    raw_calls_df = (
        raw_day_calls_df
            .union(raw_eve_calls_df)
            .union(raw_intl_calls_df)
            .union(raw_night_calls_df)
    )

    calls_df = (
        raw_calls_df
            .groupBy("User_id")
            .pivot("Call_type", ["Day", "Eve", "Night", "Intl"])
            .agg(F.sum("Duration").alias("Mins"), F.count("Duration").alias("Calls"))
    )

    # Join subscribers with calls
    joined_df = subscribers_df.join(calls_df, on=['User_id'], how='left')
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
    joined_df = joined_df.drop("Area_Code")

    for state in STATES:
        joined_df = joined_df.withColumn(
            "State_{}".format(state),
            F.when(F.col("State") == state, 1).otherwise(0)
        )
    joined_df = joined_df.drop("State")

    joined_df = joined_df.drop("User_id")
    return joined_df
