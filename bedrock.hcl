 // Refer to https://docs.basis-ai.com/getting-started/writing-files/bedrock.hcl for more details.
version = "1.0"

/*
Train stanza
Comprises the following:
- [required] step: training steps to be run. Multiple steps are allowed but must have different names
- [optional] parameters: environment variables used by the script. They can be overwritten when you create a run.
- [optional] secrets: the names of the secrets necessary to run the script successfully

Step stanza
Comprises the following:
- [required] image: the base Docker image that the script will run in
- [optional] install: the command to install any other packages not covered in the image
- [required] script: the command that calls the script
- [optional] resources: the computing resources to be allocated to this run step
- [optional] depends_on: a list of names of steps that this run step depends on
*/
train {
    // We declare a step with a step name. For example, this step is named as "preprocess".
    // A step's name must be unique.
    step preprocess {
        image = "basisai/workload-standard:v0.2.1"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        // As we are using Spark, "script" is written in the manner shown below.
        // If Spark is not required, it is just simply:
        // script = [{sh = ["python3 train.py"]}]
        script = [
            {spark-submit {
                script = "preprocess.py"
                // to be passed in as --conf key=value
                conf {
                    spark.kubernetes.container.image = "basisai/workload-standard:v0.2.1"
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
                // to be passed in as --key=value
                settings {
                }
            }}
        ]
        resources {
            cpu = "0.5"
            memory = "1G"
        }
    }

    step generate_features {
        image = "basisai/workload-standard:v0.2.1"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [
            {spark-submit {
                script = "generate_features.py"
                conf {
                    spark.kubernetes.container.image = "basisai/workload-standard:v0.2.1"
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
        image = "basisai/workload-standard:v0.2.1"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [{sh = ["python3 train.py"]}]
        resources {
            cpu = "0.5"
            memory = "1G"
            // gpu = "1"  // uncomment this in order to use GPU. Only integer values are allowed.
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

    // only provide the NAMES of the secrets here, NOT the secret values.
    // you will enter the secret values from Bedrock web UI.
    /*
    secrets = [
        "SECRET_KEY_1",
        "SECRET_KEY_2"
    ]
    */
}

/*
Batch score stanza
Similar in style as Train stanza
*/
batch_score {
    step preprocess {
        image = "basisai/workload-standard:v0.2.1"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [
            {spark-submit {
                script = "preprocess.py"
                conf {
                    spark.kubernetes.container.image = "basisai/workload-standard:v0.2.1"
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
        image = "basisai/workload-standard:v0.2.1"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [
            {spark-submit {
                script = "generate_features.py"
                conf {
                    spark.kubernetes.container.image = "basisai/workload-standard:v0.2.1"
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

    step batch_score {
        image = "basisai/workload-standard:v0.2.1"
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
        RAW_SUBSCRIBERS_DATA = "s3a://bedrock-sample/churn_data/subscribers.gz.parquet"
        RAW_CALLS_DATA = "s3a://bedrock-sample/churn_data/all_calls.gz.parquet"
        TEMP_DATA_BUCKET = "s3a://span-production-temp-data/"
        PREPROCESSED_DATA = "churn_data/preprocessed"
        FEATURES_DATA = "churn_data/features.csv"
        BIGQUERY_PROJECT = "span-production"
        BIGQUERY_DATASET = "churn"
        DEST_SUBSCRIBER_SCORE_TABLE = "subscriber_score"
        OUTPUT_MODEL_NAME = "lgb_model.pkl"
    }
}

/*
Serve stanza for HTTP serving
Only comprises the following:
- [required] image: the base Docker image that the script will run in
- [optional] install: the command to install any other packages not covered in the image
- [required] script: the command that calls the script
*/
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
