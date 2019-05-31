"""
Script to preprocess subscriber data and save to feature store.
"""
import os
import time
from bedrock_client.bedrock.feature_store import get_feature_store
from utils.preprocess import preprocess_subscriber
from pyspark.sql import SparkSession

SUBSCRIBER_FS = os.getenv("SUBSCRIBER_FS")


if __name__ == '__main__':
    with SparkSession.builder.appName("SubscriberPreprocessing").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("FATAL")

        start = time.time()
        print("\tLoading active subscribers")
        subscriber_df = preprocess_subscriber(spark)
        subscriber_pandasdf = (
            subscriber_df
            .filter(subscriber_df["Churn"] == 0)
            .drop("Churn")
            .toPandas()
        )
        print("\tTime taken = {:.2f} min".format((time.time() - start) / 60))
        print("\tNumber of active subscribers = {}".format(subscriber_pandasdf.shape[0]))

    fs = get_feature_store()
    start = time.time()
    print("\tWriting active subscribers into fs")
    fs.write_pandas_df(subscriber_pandasdf, SUBSCRIBER_FS, "User_id")
    print("\tTime taken = {:.2f} min".format((time.time() - start) / 60))

    print("\tTest calling feature store")
    User_id = "68b1fbe7f16e4ae3024973f12f3cb313"
    print(fs.read(SUBSCRIBER_FS, [User_id])[User_id])
