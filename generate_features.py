"""
Script to generate features.
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from utils.constants import AREA_CODES, STATES, FEATURE_COLS, TARGET_COL, USER_COL

PREPROCESSED_DATA = os.path.join(os.getenv("TEMP_DATA_BUCKET"),
                                 os.getenv("PREPROCESSED_DATA"))
FEATURES_DATA = os.path.join(os.getenv("TEMP_DATA_BUCKET"),
                             os.getenv("FEATURES_DATA"))


def generate_features(spark):
    """Generate features."""
    joined_df = spark.read.parquet(PREPROCESSED_DATA)
    for area_code in AREA_CODES:
        joined_df = joined_df.withColumn(
            "Area_Code={}".format(area_code),
            F.when(F.col("Area_Code") == area_code, 1).otherwise(0)
        )

    for state in STATES:
        joined_df = joined_df.withColumn(
            "State={}".format(state),
            F.when(F.col("State") == state, 1).otherwise(0)
        )

    joined_df = joined_df.select(FEATURE_COLS + [TARGET_COL] + [USER_COL])
    return joined_df


def main():
    """Generate features"""
    print("\tGenerating features")
    with SparkSession.builder.appName("FeatureGeneration").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("FATAL")
        features_data = generate_features(spark).toPandas()
        features_data.to_csv(FEATURES_DATA, index=False)


if __name__ == "__main__":
    main()
