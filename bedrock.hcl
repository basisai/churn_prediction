version = "1.0"

/*
Train stanza
Comprises the following:
- [required] image: the base Docker image that the script will run in
- [optional] install: the command to install any other packages not covered in the image
- [required] script: the command that calls the script
- [optional] parameters: any environment variables used by the script for convenience
- [optional] secrets: the names of the secrets necessary to run the script successfully

Please refer to https://docs.basis-ai.com/getting-started/writing-files/bedrock.hcl for more details.
*/
train {
    image = "basisai/workload-standard:v0.1.2"
    install = ["pip3 install -r requirements.txt"]
    // As we are using Spark, "script" is written in the manner shown below.
    // If Spark is not required, it is just simply:
    // script = [{sh = ["python3 train.py"]}]
    script = [
        {spark-submit {
            script = "train.py"
            // to be passed in as --conf key=value
            conf {
                spark.kubernetes.container.image = "basisai/workload-standard:v0.1.2"
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
        N_ESTIMATORS = "150"
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
    image = "basisai/workload-standard:v0.1.2"
    install = ["pip3 install -r requirements.txt && pip3 install pandas-gbq"]
    script = [
        {spark-submit {
            script = "batch_score.py"
            // to be passed in as --conf key=value
            conf {
                spark.kubernetes.container.image = "basisai/workload-standard:v0.1.2"
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
        BIGQUERY_PROJECT = "span-production"
        BIGQUERY_DATASET = "churn"
        DEST_SUBSCRIBER_SCORE_TABLE = "subscriber_score"
        OUTPUT_MODEL_NAME = "lgb_model.pkl"
    }
}

/*
Serve stanza
Only comprises the following:
- [required] image: the base Docker image that the script will run in
- [optional] install: the command to install any other packages not covered in the image
- [required] script: the command that calls the script
*/
serve {
    image = "python:3.7"
    install = ["pip3 install -r requirements-serve.txt"]
    script = ["python3 serve_http.py"]
}
