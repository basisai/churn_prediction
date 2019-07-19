"""
Script for serving.
"""
import json
import os
import pickle
import socketserver
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler

import numpy as np

from utils.constants import AREA_CODES, STATES, SUBSCRIBER_FEATURES

OUTPUT_MODEL_NAME = "lgb_model.pkl"
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8080"))


class Handler(SimpleHTTPRequestHandler):
    def predict_prob(self, subscriber_features):
        """Predict churn probability given subscriber_features.

        Args:
            subscriber_features (dict)

        Returns:
            churn_prob (float): churn probability
        """
        with open(OUTPUT_MODEL_NAME, "rb") as model_file:
            model = pickle.load(model_file)

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
        churn_prob = model.predict_proba(np.array(row_feats).reshape(1, -1))[:, 1].item()
        return churn_prob

    def do_GET(self):
        # self.send_response(HTTPStatus.OK)
        # self.end_headers()
        # self.wfile.write(b'Hello, world!')
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        self.send_response(HTTPStatus.OK)
        self.end_headers()

        subscriber_features = json.loads(post_data.decode("utf-8"))
        result = {
            "churn_prob": self.predict_prob(subscriber_features)
        }
        self.wfile.write(bytes(json.dumps(result).encode('utf-8')))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        self.send_response(HTTPStatus.OK)
        self.end_headers()

        subscriber_features = json.loads(post_data.decode("utf-8"))
        result = {
            "churn_prob": self.predict_prob(subscriber_features)
        }
        self.wfile.write(bytes(json.dumps(result).encode('utf-8')))


def main():
    print(f"Starting server at {SERVER_PORT}")
    httpd = socketserver.TCPServer(("", SERVER_PORT), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
