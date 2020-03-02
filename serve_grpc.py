"""
Script for serving.
"""
from concurrent.futures import ThreadPoolExecutor
import os
import logging
import pickle
import time

import grpc
import numpy as np
import serve_pb2
import serve_pb2_grpc

from utils.constants import AREA_CODES, STATES, SUBSCRIBER_FEATURES

OUTPUT_MODEL_NAME = "/artefact/lgb_model.pkl"
SERVER_PORT = os.environ.get("BEDROCK_SERVER_PORT", "50051")

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class PredictorService(serve_pb2_grpc.PredictorServicer):
    """grpc predictor service"""
    def __init__(self):
        with open(OUTPUT_MODEL_NAME, "rb") as model_file:
            self.model = pickle.load(model_file)

    def predict_prob(self, subscriber_features):
        """Predict churn probability of user with User_id.

        Args:
            subscriber_features (dict)

        Returns:
            churn_prob (float): churn probability
        """
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
        churn_prob = (
            self.model
            .predict_proba(np.array(row_feats).reshape(1, -1))[:, 1].item()
        )
        return churn_prob

    # pylint: disable=unused-argument
    def PredictProb(self, request, context):
        """Entry point for predicting churn probability."""
        # pylint: disable=broad-except,invalid-name
        try:
            churn_prob = self.predict_prob(
                {
                    "State": request.state,
                    "Area_Code": request.area_code,
                    "Intl_Plan": request.intl_plan,
                    "VMail_Plan": request.vmail_plan,
                    "VMail_Message": request.vmail_message,
                    "CustServ_Calls": request.custserv_calls,
                    "Day_Mins": request.day_mins,
                    "Day_Calls": request.day_calls,
                    "Eve_Mins": request.eve_mins,
                    "Eve_Calls": request.eve_calls,
                    "Night_Mins": request.night_mins,
                    "Night_Calls": request.night_calls,
                    "Intl_Mins": request.intl_mins,
                    "Intl_Calls": request.intl_calls,
                }
            )
            return serve_pb2.PredictResponse(churn_prob=churn_prob)
        except Exception as e:
            logging.error(e, exc_info=True)


def main():
    """Start grpc server"""
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    serve_pb2_grpc.add_PredictorServicer_to_server(PredictorService(), server)
    server.add_insecure_port(f"[::]:{SERVER_PORT}")
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    main()
