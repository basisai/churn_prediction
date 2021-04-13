version = "1.0"

train {
    step preprocess {
        image = "basisai/workload-standard:v0.2.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements-aws.txt",
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
                    spark.hadoop.fs.s3a.impl = "org.apache.hadoop.fs.s3a.S3AFileSystem"
                    spark.hadoop.fs.s3a.endpoint = "s3.ap-southeast-1.amazonaws.com"
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
            "pip3 install -r requirements-aws.txt",
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
                    spark.hadoop.fs.s3a.impl = "org.apache.hadoop.fs.s3a.S3AFileSystem"
                    spark.hadoop.fs.s3a.endpoint = "s3.ap-southeast-1.amazonaws.com"
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
            "pip3 install -r requirements-aws.txt",
        ]
        script = [{sh = ["python3 train.py"]}]
        resources {
            cpu = "0.5"
            memory = "1G"
        }
        depends_on = ["generate_features"]
    }

    parameters {
        RAW_SUBSCRIBERS_DATA = "s3a://bedrock-sample/churn_data/subscribers.gz.parquet"
        RAW_CALLS_DATA = "s3a://bedrock-sample/churn_data/all_calls.gz.parquet"
        TEMP_DATA_BUCKET = "s3a://span-production-temp-data/"
        PREPROCESSED_DATA = "churn_data/preprocessed"
        FEATURES_DATA = "churn_data/features.csv"
        LR = "0.05"
        NUM_LEAVES = "10"
        N_ESTIMATORS = "100"
        OUTPUT_MODEL_NAME = "lgb_model.pkl"
    }
}

serve {
    image = "python:3.7"
    install = [
        "pip3 install --upgrade pip",
        "pip3 install -r requirements-serve.txt",
    ]
    script = [
        {sh = [
            "gunicorn --config gunicorn_config.py --bind=:${BEDROCK_SERVER_PORT:-8080} --worker-class=gthread --workers=${WORKERS} --timeout=300 --preload serve_http:app"
        ]}
    ]

    parameters {
        WORKERS = "2"
        prometheus_multiproc_dir = "/tmp"
    }
}
