"""
Script for serving.
"""
import os
import logging
import pickle
import time
import grpc
import numpy as np
import lightgbm as lgb

import serve_pb2
import serve_pb2_grpc
from bedrock_client.bedrock.feature_store import get_feature_store
from concurrent.futures import ThreadPoolExecutor
from utils.constants import AREA_CODES, STATES, SUBSCRIBER_FEATURES

SUBSCRIBER_FS = "subscribers"
OUTPUT_MODEL_NAME = "lgb_model.pkl"

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class PredictorService(serve_pb2_grpc.PredictorServicer):
    def __init__(self):
        self.feature_store = get_feature_store()
        with open(OUTPUT_MODEL_NAME, "rb") as model_file:
            self.model: lgb = pickle.load(model_file)
        # self.model._make_predict_function()

    def predict_prob(self, User_id):
        """Predict churn probability of user with User_id.

        Args:
            User_id (str)

        Returns:
            churn_prob (float): churn probability
        """
        # Get subscriber features from feature store
        subscriber_features = self.feature_store.read(
            SUBSCRIBER_FS, [User_id])[User_id]

        row_feats = list()
        for col in SUBSCRIBER_FEATURES:
            row_feats.append(subscriber_features[col])

        for area_code in AREA_CODES:
            if subscriber_features["Area_Code"] == area_code:
                row_feats.append(1)
            else:
                row_feats.append(0)

        for state in STATES:
            if subscriber_features["State"] == state:
                row_feats.append(1)
            else:
                row_feats.append(0)

        # Score
        churn_prob = self.model.predict(np.array(row_feats).reshape(1, -1))[:, 1].item()
        return churn_prob

    def PredictProb(self, request, context):
        try:
            churn_prob = self.predict_prob(request.User_id)
            return serve_pb2.PredictResponse(churn_prob=churn_prob)
        except Exception as e:
            logging.error(e, exc_info=True)


if __name__ == "__main__":
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    serve_pb2_grpc.add_PredictorServicer_to_server(PredictorService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
