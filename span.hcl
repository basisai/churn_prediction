version = "1.0"

train {
    image = "asia.gcr.io/span-ai/ds_suite-pyspark:v0.11"
    install = ["pip3 install -r requirements.txt"]
    script = ["train.py"]

    parameters {
        RAW_BIGQUERY_PROJECT = "span-production"
        RAW_BIGQUERY_DATASET = "churn"
        RAW_SUBSCRIBER_TABLE = "subscribers"
        RAW_DAY_CALL_TABLE = "Day_calls"
        RAW_EVE_CALL_TABLE = "Eve_calls"
        RAW_INTL_CALL_TABLE = "Intl_calls"
        RAW_NIGHT_CALL_TABLE = "Night_calls"
        LR = "0.05"
        NUM_LEAVES = "10"
        N_ESTIMATORS = "250"
        OUTPUT_MODEL_NAME = "lgb_model.pkl"
    }

    spark {
        // to be passed in as --conf key=value
        conf {
            spark.executor.instances = "2"
            spark.kubernetes.container.image = "asia.gcr.io/span-ai/ds_suite-pyspark:v0.11"
            spark.kubernetes.pyspark.pythonVersion = "3"
            spark.driver.memory = "4g"
            spark.driver.cores = "2"
            spark.executor.memory = "4g"
            spark.executor.cores = "2"
            spark.memory.fraction = "0.5"
            spark.hadoop.fs.AbstractFileSystem.gs.impl = "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS"
            spark.hadoop.google.cloud.auth.service.account.enable = "true"
            spark.sql.parquet.compression.codec = "gzip"
        }
        // to be passed in as --key=value
        settings {
            jars = "gs://spark-lib/bigquery/spark-bigquery-latest.jar"
        }
    }
}

features {
    image = "asia.gcr.io/span-ai/ds_suite-pyspark:v0.11"
    install = ["pip3 install -r requirements.txt"]
    script = ["fs_preprocess.py"]

    parameters {
        RAW_BIGQUERY_PROJECT = "span-production"
        RAW_BIGQUERY_DATASET = "churn"
        RAW_SUBSCRIBER_TABLE = "subscribers"
        RAW_DAY_CALL_TABLE = "Day_calls"
        RAW_EVE_CALL_TABLE = "Eve_calls"
        RAW_INTL_CALL_TABLE = "Intl_calls"
        RAW_NIGHT_CALL_TABLE = "Night_calls"
        SUBSCRIBER_FS = "subscriber_fs"
    }

    spark {
        // to be passed in as --conf key=value
        conf {
            spark.executor.instances = "2"
            spark.kubernetes.container.image = "asia.gcr.io/span-ai/ds_suite-pyspark:v0.11"
            spark.kubernetes.pyspark.pythonVersion = "3"
            spark.driver.memory = "4g"
            spark.driver.cores = "2"
            spark.executor.memory = "4g"
            spark.executor.cores = "2"
            spark.memory.fraction = "0.5"
            spark.hadoop.fs.AbstractFileSystem.gs.impl = "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS"
            spark.hadoop.google.cloud.auth.service.account.enable = "true"
            spark.sql.parquet.compression.codec = "gzip"
        }
        // to be passed in as --key=value
        settings {
            jars = "gs://spark-lib/bigquery/spark-bigquery-latest.jar"
        }
    }
    
    feature_definition = [
        {
          name = "subscribers_fs"
          key = "User_id"
          description = "Contains details of all active subscribers. User_id (str)"
        }
    ]
}

serve {
    image = "asia.gcr.io/span-ai/ds_suite-pyspark:v0.11"
    install = [
        "pip install -r requirements.txt && pip install google-cloud-storage grpcio-tools grpcio protobuf",
        "python3 -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/serve.proto"
    ]
    script = ["python3 serve.py"]
}
