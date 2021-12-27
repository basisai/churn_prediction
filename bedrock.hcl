version = "1.0"

train {
  step "preprocess" {
    # Same as spark.kubernetes.container.image
    image = "quay.io/basisai/workload-standard:v0.3.4"
    install = []
    script = [
      {
        spark-submit = {
          script = "preprocess.py"
          conf = {
            "spark.executor.instances"               = "2"
            "spark.executor.memory"                  = "4g"
            "spark.executor.cores"                   = "2"
            "spark.sql.parquet.compression.codec"    = "gzip"
          }
        }
      }
    ]
    resources {
      # Same as spark.driver.cores
      cpu    = "0.5"
      # Same as spark.driver.memory
      memory = "1G"
    }
    retry {
      limit = 2
    }
  }

  step "generate_features" {
    image = "quay.io/basisai/workload-standard:v0.3.4"
    install = [
      "pip3 install --upgrade pip",
      "pip3 install -r requirements.txt",
    ]
    script = [
      {
        spark-submit = {
          script = "generate_features.py"
          conf = {
            "spark.executor.instances"               = "2"
            "spark.executor.memory"                  = "4g"
            "spark.executor.cores"                   = "2"
            "spark.sql.parquet.compression.codec"    = "gzip"
          }
        }
      }
    ]
    resources {
      cpu    = "0.5"
      memory = "1G"
    }
    retry {
      limit = 2
    }
    depends_on = ["preprocess"]
  }

  step "train" {
    image = "python:3.7"
    install = [
      "pip3 install --upgrade pip",
      "pip3 install -r requirements.txt",
    ]
    script = [{ sh = ["python3 train.py"] }]
    resources {
      cpu    = "0.5"
      memory = "1G"
    }
    retry {
      limit = 2
    }
    depends_on = ["generate_features"]
  }

  parameters {
    RAW_SUBSCRIBERS_DATA = "s3a://bedrock-sample/churn_data/subscribers.gz.parquet"
    RAW_CALLS_DATA       = "s3a://bedrock-sample/churn_data/all_calls.gz.parquet"
    TEMP_DATA_BUCKET     = "s3a://span-production-temp-data/"
    PREPROCESSED_DATA    = "churn_data/preprocessed"
    FEATURES_DATA        = "churn_data/features.csv"
    LR                   = "0.05"
    NUM_LEAVES           = "10"
    N_ESTIMATORS         = "100"
    OUTPUT_MODEL_NAME    = "lgb_model.pkl"
  }
}

serve {
  image = "python:3.7"
  install = [
    "pip3 install --upgrade pip",
    "pip3 install -r requirements-serve.txt",
  ]
  script = [
    {
      sh = [
        "gunicorn --config gunicorn_config.py --bind=:${BEDROCK_SERVER_PORT:-8080} --worker-class=gthread --workers=${WORKERS} --timeout=300 --preload serve_http:app"
      ]
    }
  ]

  parameters {
    WORKERS                  = "2"
    PROMETHEUS_MULTIPROC_DIR = "/tmp"
  }
}

batch_score {
  step "preprocess" {
    image = "quay.io/basisai/workload-standard:v0.3.4"
    install = []
    script = [
      {
        spark-submit = {
          script = "preprocess.py"
          conf = {
            "spark.executor.instances"               = "2"
            "spark.executor.memory"                  = "4g"
            "spark.executor.cores"                   = "2"
            "spark.sql.parquet.compression.codec"    = "gzip"
          }
        }
      }
    ]
    resources {
      cpu    = "0.5"
      memory = "1G"
    }
    retry {
      limit = 2
    }
  }

  step "generate_features" {
    image = "quay.io/basisai/workload-standard:v0.3.4"
    install = [
      "pip3 install --upgrade pip",
      "pip3 install -r requirements.txt",
    ]
    script = [
      {
        spark-submit = {
          script = "generate_features.py"
          conf = {
            "spark.executor.instances"               = "2"
            "spark.executor.memory"                  = "4g"
            "spark.executor.cores"                   = "2"
            "spark.sql.parquet.compression.codec"    = "gzip"
          }
        }
      }
    ]
    resources {
      cpu    = "0.5"
      memory = "1G"
    }
    retry {
      limit = 2
    }
    depends_on = ["preprocess"]
  }

  step "batch_score" {
    image = "python:3.7"
    install = [
      "pip3 install --upgrade pip",
      "pip3 install -r requirements.txt",
    ]
    script = [{ sh = ["python3 batch_score.py"] }]
    resources {
      cpu    = "0.5"
      memory = "1G"
    }
    retry {
      limit = 2
    }
    depends_on = ["generate_features"]
  }

  parameters {
    RAW_SUBSCRIBERS_DATA  = "s3a://bedrock-sample/churn_data/subscribers.gz.parquet"
    RAW_CALLS_DATA        = "s3a://bedrock-sample/churn_data/all_calls.gz.parquet"
    TEMP_DATA_BUCKET      = "s3a://span-production-temp-data/"
    PREPROCESSED_DATA     = "churn_data/preprocessed"
    FEATURES_DATA         = "churn_data/features.csv"
    SUBSCRIBER_SCORE_DATA = "churn_data/subscriber_score.csv"
    OUTPUT_MODEL_NAME     = "lgb_model.pkl"
  }
}
