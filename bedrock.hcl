version = "1.0"

train {
    image = "basisai/workload-standard:v0.1.2"
    install = ["pip3 install -r requirements.txt"]
    script = [
        {spark-submit {
            script = "train.py"
            // to be passed in as --conf key=value
            conf {
                spark.kubernetes.container.image = "basisai/workload-standard:v0.1.0"
                spark.kubernetes.pyspark.pythonVersion = "3"
                spark.driver.memory = "4g"
                spark.driver.cores = "2"
                spark.executor.instances = "2"
                spark.executor.memory = "4g"
                spark.executor.cores = "2"
                spark.memory.fraction = "0.5"
                spark.sql.parquet.compression.codec = "gzip"
                spark.hadoop.fs.AbstractFileSystem.gs.impl = "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS"
                spark.hadoop.google.cloud.auth.service.account.enable = "true"
            }
            // to be passed in as --key=value
            settings {
            }
        }}
    ]

    parameters {
        RAW_SUBSCRIBERS_DATA = "gs://bedrock-sample/churn_data/subscribers.gz.parquet"
        RAW_CALLS_DATA = "gs://bedrock-sample/churn_data/all_calls.gz.parquet"
        LR = "0.05"
        NUM_LEAVES = "10"
        N_ESTIMATORS = "250"
        OUTPUT_MODEL_NAME = "lgb_model.pkl"
    }
}

batch_score {
    image = "basisai/workload-standard:v0.1.2"
    install = ["pip3 install -r requirements.txt && pip3 install pandas-gbq"]
    script = [
        {spark-submit {
            script = "batch_score.py"
            // to be passed in as --conf key=value
            conf {
                spark.kubernetes.container.image = "basisai/workload-standard:v0.1.0"
                spark.kubernetes.pyspark.pythonVersion = "3"
                spark.driver.memory = "4g"
                spark.driver.cores = "2"
                spark.executor.instances = "2"
                spark.executor.memory = "4g"
                spark.executor.cores = "2"
                spark.memory.fraction = "0.5"
                spark.sql.parquet.compression.codec = "gzip"
                spark.hadoop.fs.AbstractFileSystem.gs.impl = "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS"
                spark.hadoop.google.cloud.auth.service.account.enable = "true"
            }
            // to be passed in as --key=value
            settings {
                jars = "gs://spark-lib/bigquery/spark-bigquery-latest.jar"
            }
        }}
    ]

    parameters {
        RAW_SUBSCRIBERS_DATA = "gs://bedrock-sample/churn_data/subscribers.gz.parquet"
        RAW_CALLS_DATA = "gs://bedrock-sample/churn_data/all_calls.gz.parquet"
        OUTPUT_MODEL_NAME = "lgb_model.pkl"
        DEST_BIGQUERY_PROJECT = "span-production"
        DEST_BIGQUERY_DATASET = "churn"
        DEST_SUBSCRIBER_SCORE_TABLE = "subscriber_score"

    }
}

serve {
    image = "python:3.7"
    install = [
        "pip3 install bdrk==0.0.1 numpy==1.17.0 lightgbm==2.2.3 grpcio-tools==1.18.0 grpcio==1.18.0 protobuf==3.6.1",
        "python3 -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/serve.proto"
    ]
    script = ["python3 serve_grpc.py"]
}
