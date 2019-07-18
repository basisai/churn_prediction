"""
Script to preprocess subscriber data and save to feature store.
"""
import os
import time

from bedrock_client.bedrock.feature_store import get_feature_store
from pyspark.sql import SparkSession

from utils.preprocess import preprocess_subscriber


SUBSCRIBER_FS = os.getenv("SUBSCRIBER_FS")


def main():
    """Feature pipeline"""
    with (SparkSession.builder
          .appName("SubscriberPreprocessing")
          .getOrCreate()) as spark:
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
        print("\tNumber of active subscribers = {}"
              .format(subscriber_pandasdf.shape[0]))

    feature_store = get_feature_store()
    start = time.time()
    print("\tWriting active subscribers into fs")
    feature_store.write_pandas_df(subscriber_pandasdf,
                                  SUBSCRIBER_FS,
                                  "User_id")
    print("\tTime taken = {:.2f} min".format((time.time() - start) / 60))

    print("\tTest calling feature store")
    user_id = "68b1fbe7f16e4ae3024973f12f3cb313"
    print(feature_store.read(SUBSCRIBER_FS, [user_id])[user_id])


if __name__ == '__main__':
    main()
