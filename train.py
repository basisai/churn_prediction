"""
Script to train model.
"""
import logging
import os
import pickle
import sys
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from utils.constants import AREA_CODES, STATES, SUBSCRIBER_FEATURES
from utils.preprocess import generate_features
from pyspark.sql import SparkSession

from bedrock_client.bedrock.api import BedrockApi

LR = float(os.getenv('LR'))
NUM_LEAVES = int(os.getenv('NUM_LEAVES'))
N_ESTIMATORS = int(os.getenv('N_ESTIMATORS'))
OUTPUT_MODEL_NAME = os.getenv('OUTPUT_MODEL_NAME')


if __name__ == '__main__':
    print("\tGenerate features")
    with SparkSession.builder.appName("Preprocessing").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("FATAL")
        model_data = generate_features(spark).toPandas()

    print("\tSplit train and validation data")
    TARGET_COL = "Churn"
    FEATURE_COLS = SUBSCRIBER_FEATURES + \
        [f"Area_Code_{area_code}" for area_code in AREA_CODES] + \
        [f"State_{state}" for state in STATES]

    X = model_data[FEATURE_COLS].values
    y = model_data[TARGET_COL].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("\tTrain model")
    gbm = lgb.LGBMClassifier(
        num_leaves=NUM_LEAVES,
        learning_rate=LR,
        n_estimators=N_ESTIMATORS,
    )
    gbm.fit(X_train, y_train)

    print("\tSave model")
    with open("/artefact/" + OUTPUT_MODEL_NAME, "wb") as model_file:
        pickle.dump(gbm, model_file)

    print("\tPredict using validation data")
    y_prob = gbm.predict_proba(X_val)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    acc = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    auc = metrics.roc_auc_score(y_val, y_prob)
    avg_prc = metrics.average_precision_score(y_val, y_prob)
    print("Accuracy =", acc)
    print("Precision =", precision)
    print("Recall =", recall)
    print("F1 score =", f1_score)
    print("AUC =", auc)
    print("Average precision =", avg_prc)

    # Log metrics
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s  %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    bedrock = BedrockApi(logger)
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("Precision", precision)
    bedrock.log_metric("Recall", recall)
    bedrock.log_metric("F1 score", f1_score)
    bedrock.log_metric("AUC", auc)
    bedrock.log_metric("Avg precision", avg_prc)
    bedrock.log_chart_data(y_val.astype(int).tolist(), y_prob.flatten().tolist())
