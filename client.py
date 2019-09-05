"""
Sample gRPC Client
"""
import argparse
import grpc

import serve_pb2
import serve_pb2_grpc


def do_grpc_request(channel: grpc.Channel):
    """Query given grpc channel for prediction.
    """
    client = serve_pb2_grpc.PredictorStub(channel)
    return client.PredictProb(
        serve_pb2.PredictRequest(
            state="ME",
            area_code=408,
            intl_plan=1,
            vmail_plan=1,
            vmail_message=21,
            custserv_calls=4,
            day_mins=156.5,
            day_calls=122,
            eve_mins=209.2,
            eve_calls=125,
            night_mins=158.7,
            night_calls=81,
            intl_mins=11.1,
            intl_calls=3,
        )
    )


def main():
    """Start gRPC Client"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--endpoint", type=str, required=True, help="Target endpoint"
    )

    args = parser.parse_args()
    endpoint = args.endpoint

    credentials = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel(endpoint, credentials)
    print(do_grpc_request(channel))


if __name__ == "__main__":
    main()
