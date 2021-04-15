version = "1.0"

train {
    step preprocess {
        image = "basisai/workload-standard:v0.2.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [
            {spark-submit {
                script = "preprocess.py"
                conf {
                    spark.kubernetes.container.image = "basisai/workload-standard:v0.2.2"
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
            }}
        ]
        resources {
            cpu = "0.5"
            memory = "1G"
        }
    }

    step generate_features {
        image = "basisai/workload-standard:v0.2.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [
            {spark-submit {
                script = "generate_features.py"
                conf {
                    spark.kubernetes.container.image = "basisai/workload-standard:v0.2.2"
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
            }}
        ]
        resources {
            cpu = "0.5"
            memory = "1G"
        }
        depends_on = ["preprocess"]
    }

    step train {
        image = "basisai/workload-standard:v0.2.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [{sh = ["python3 train.py"]}]
        resources {
            cpu = "0.5"
            memory = "1G"
        }
        depends_on = ["generate_features"]
    }

    parameters {
        RAW_SUBSCRIBERS_DATA = "gs://bedrock-sample/churn_data/subscribers.gz.parquet"
        RAW_CALLS_DATA = "gs://bedrock-sample/churn_data/all_calls.gz.parquet"
        TEMP_DATA_BUCKET = "gs://span-temp-production/"
        PREPROCESSED_DATA = "churn_data/preprocessed"
        FEATURES_DATA = "churn_data/features.csv"
        LR = "0.05"
        NUM_LEAVES = "10"
        N_ESTIMATORS = "100"
        OUTPUT_MODEL_NAME = "lgb_model.pkl"
    }

    // only provide the NAMES of the secrets here, NOT the secret values.
    // you will enter the secret values from Bedrock web UI.
    /*
    secrets = [
        "SECRET_KEY_1",
        "SECRET_KEY_2"
    ]
    */
}

serve {
    image = "python:3.7"
    // The third line in "install" is to generate serve_pb2 and serve_pb2_grpc.
    // It requires protos/serve.proto.
    install = [
        "pip3 install --upgrade pip",
        "pip3 install -r requirements-serve.txt",
        "python3 -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/serve.proto",
    ]
    script = [{sh = ["python3 serve_grpc.py"]}]

    parameters {
        WORKERS = "2"
    }
}

batch_score {
    step preprocess {
        image = "basisai/workload-standard:v0.2.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [
            {spark-submit {
                script = "preprocess.py"
                conf {
                    spark.kubernetes.container.image = "basisai/workload-standard:v0.2.2"
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
            }}
        ]
        resources {
            cpu = "0.5"
            memory = "1G"
        }
    }

    step generate_features {
        image = "basisai/workload-standard:v0.2.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [
            {spark-submit {
                script = "generate_features.py"
                conf {
                    spark.kubernetes.container.image = "basisai/workload-standard:v0.2.2"
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
            }}
        ]
        resources {
            cpu = "0.5"
            memory = "1G"
        }
        depends_on = ["preprocess"]
    }

    step batch_score {
        image = "basisai/workload-standard:v0.2.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
            "pip3 install pandas-gbq",
        ]
        script = [{sh = ["python3 batch_score.py"]}]
        resources {
            cpu = "0.5"
            memory = "1G"
        }
        depends_on = ["generate_features"]
    }

    parameters {
        RAW_SUBSCRIBERS_DATA = "gs://bedrock-sample/churn_data/subscribers.gz.parquet"
        RAW_CALLS_DATA = "gs://bedrock-sample/churn_data/all_calls.gz.parquet"
        TEMP_DATA_BUCKET = "gs://span-temp-production/"
        PREPROCESSED_DATA = "churn_data/preprocessed"
        FEATURES_DATA = "churn_data/features.csv"
        BIGQUERY_PROJECT = "span-production"
        BIGQUERY_DATASET = "churn"
        SUBSCRIBER_SCORE_TABLE = "subscriber_score"
        OUTPUT_MODEL_NAME = "lgb_model.pkl"
    }
}
