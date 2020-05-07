"""
Script to preprocess subscribers.
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

RAW_SUBSCRIBERS_DATA = os.getenv("RAW_SUBSCRIBERS_DATA")
RAW_CALLS_DATA = os.getenv("RAW_CALLS_DATA")
PREPROCESSED_DATA = os.path.join(os.getenv("TEMP_DATA_BUCKET"),
                                 os.getenv("PREPROCESSED_DATA"))


def preprocess_subscriber(spark):
    """Preprocess subscriber data."""
    # Load subscribers
    subscribers_df = (
        spark.read.parquet(RAW_SUBSCRIBERS_DATA)
        .withColumn("Intl_Plan", F.when(F.col("Intl_Plan") == "yes", 1).otherwise(0))
        .withColumn("VMail_Plan", F.when(F.col("VMail_Plan") == "yes", 1).otherwise(0))
        .withColumn("Churn", F.when(F.col("Churn") == "yes", 1).otherwise(0))
    )

    # Load raw calls
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


def main():
    """Preprocess data"""
    print("\tPreprocessing")
    with SparkSession.builder.appName("Preprocessing").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("FATAL")
        preprocessed_df = preprocess_subscriber(spark)
        preprocessed_df.repartition(1).write.mode("overwrite").parquet(PREPROCESSED_DATA)


if __name__ == "__main__":
    main()
