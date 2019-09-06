"""
Script to preprocess subscribers.
"""
import os
from .constants import AREA_CODES, STATES
from pyspark.sql import functions as F

RAW_SUBSCRIBERS_DATA = os.getenv("RAW_SUBSCRIBERS_DATA")
RAW_CALLS_DATA = os.getenv("RAW_CALLS_DATA")


def preprocess_subscriber(spark):
    """Preprocess subscriber data."""
    # Load subscribers
    subscribers_df = (
        spark.read.parquet(RAW_SUBSCRIBERS_DATA)
        .withColumn("Intl_Plan", F.when(F.col("Intl_Plan") == "yes", 1).otherwise(0))
        .withColumn("VMail_Plan", F.when(F.col("VMail_Plan") == "yes", 1).otherwise(0))
        .withColumn("Churn", F.when(F.isnull("Date_Closed"), 0).otherwise(1))
        .drop("Date_Created")
        .drop("Date_Closed")
        .drop("Phone")
    )

    calls_df = (
        spark.read.parquet(RAW_CALLS_DATA)
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
    joined_df = joined_df.drop("Area_Code")

    for state in STATES:
        joined_df = joined_df.withColumn(
            "State_{}".format(state),
            F.when(F.col("State") == state, 1).otherwise(0)
        )
    joined_df = joined_df.drop("State")
    return joined_df
