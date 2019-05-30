import argparse
import logging
import grpc

import endpoint_profile as profile

import serve_pb2
import serve_pb2_grpc


def do_grpc_request(channel: grpc.Channel):
    """Query given grpc channel for prediction."""
    client = serve_pb2_grpc.PredictorStub(channel)
    request = serve_pb2.PredictRequest()
    request.User_id = "68b1fbe7f16e4ae3024973f12f3cb313"
    return client.PredictScore(request)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--target", type=str,
        required=True,
        help="Target endpoint (model name)"
    )
    parser.add_argument(
        "-n",
        "--num-trials",
        type=int,
        default=0,
        help="Number of data points for latency measurement if > 0, "
             "otherwise just does single query",
    )

    args = parser.parse_args()
    impressions = args.impressions
    target = args.target
    num_trials = args.num_trials

    endpoint = f"{target}.model.amoy.ai:443"

    if num_trials > 0:
        profile.measure_latency(
            grpc_request_func=do_grpc_request,
            endpoint=endpoint,
            trials=num_trials,
            repetitions=10,
        )
    else:
        profile.query(
            grpc_request_func=do_grpc_request,
            endpoint=endpoint,
        )
