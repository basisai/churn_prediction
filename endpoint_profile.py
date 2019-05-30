import time
import timeit
from datetime import datetime, timedelta
from typing import List, Tuple, Callable
import grpc
import pandas as pd

POSIX_LINE_CLEAR = "\x1b[2K"


def get_credentials() -> grpc.ChannelCredentials:
    return grpc.ssl_channel_credentials()


class ChannelManager:
    """Workaround for lack of teardown functionality in timeit.
    Maintain global list of channels that we can clear after every experiment.
    """
    channels: List[grpc.Channel] = []

    @classmethod
    def get_channel(
        cls, endpoint: str, credentials: grpc.ChannelCredentials
    ) -> grpc.Channel:
        channel = grpc.secure_channel(endpoint, credentials)
        ChannelManager.channels.append(channel)
        return channel

    @classmethod
    def clear_channels(cls):
        for channel in ChannelManager.channels:
            channel.close()
        ChannelManager.channels = []


def query(grpc_request_func: Callable, endpoint: str, **kwargs):
    """Query grpc endpoint and print results."""

    credentials = get_credentials()
    with ChannelManager.get_channel(endpoint, credentials) as channel:
        response = grpc_request_func(channel=channel, **kwargs)
        print(f"Response received: {str(response)}")


def get_measurement(
    grpc_request_func: Callable,
    setup: str,
    code: str,
    repetitions: int = 1,
    retries: int = 5,
    backoff_s: float = 1,
) -> float:
    """Wrapper around timeit module to profile code."""
    if repetitions <= 0:
        return 0.0

    # Capture these objects within the timeit call
    capture = {
        "grpc_request_func": grpc_request_func,
        "ChannelManager": ChannelManager,
        "get_credentials": get_credentials,
    }

    failures = 0
    while True:
        try:
            time_s = (
                timeit
                .Timer(code, setup=setup, globals=capture)
                .timeit(number=repetitions)
            )
            return time_s / repetitions * 1000
        except grpc._channel._Rendezvous as ex:
            failures += 1
            time.sleep(backoff_s * failures)
            if failures == retries:
                raise Exception(
                    f"ERROR: Unable to continue after multiple grpc "
                    "exceptions: {ex}"
                )


def get_latency_experiments(
    endpoint: str, **kwargs
) -> List[Tuple[str, str, str]]:
    """Get list of latency experiments:
    1. Grpc request latency including time to open new channel.
    2. Grpc request latency excluding time to open new channel.
    """

    common_setup_commands = (
        f"ChannelManager.clear_channels()\n"
        f"endpoint = '{endpoint}'\n"
        f"credentials = get_credentials()\n"
    )

    get_channel_command = (
        "channel = ChannelManager.get_channel(endpoint, credentials)\n"
    )

    kwargs_str = "".join([f",{k}={v}" for (k, v) in kwargs.items()])
    request_command = f"grpc_request_func(channel=channel{kwargs_str})\n"

    new_channel_setup = common_setup_commands
    new_channel_code = get_channel_command + request_command

    reuse_channel_setup = (
        common_setup_commands
        + get_channel_command
        + request_command  # Prime the channel
    )
    reuse_channel_code = request_command

    return [
        ("Reopen channel", new_channel_setup, new_channel_code),
        ("Reuse channel", reuse_channel_setup, reuse_channel_code),
    ]


def measure_latency(
    grpc_request_func: Callable,
    endpoint: str,
    trials: int,
    repetitions: int,
    **kwargs
):
    """Measure latency statistics for making grpc requests at given endpoint.

    :param Callable grpc_request_func: Function to execute grpc request
    :param str endpoint: Grpc endpoint
    :param int trials: Number of trials
    :param int repetitions: Number of repetitions to average over per trial
    :param kwargs: Extra args to pass to grpc request function
    """

    experiments = get_latency_experiments(endpoint=endpoint, **kwargs)

    approx_reopen_query_time_ms = 30
    approx_runtime = (
        len(experiments)
        * (trials * repetitions)
        * approx_reopen_query_time_ms / 1000
    )

    print(f"Estimated total runtime: {str(timedelta(seconds=approx_runtime))}")

    start = datetime.now()

    for experiment_name, setup, code in experiments:
        run_profiling(
            grpc_request_func=grpc_request_func,
            experiment_name=experiment_name,
            setup=setup,
            code=code,
            trials=trials,
            repetitions=repetitions,
        )
        ChannelManager.clear_channels()

    end = datetime.now()
    total_runtime = (end - start).total_seconds()
    print()
    print(f"Actual total runtime: {str(timedelta(seconds=total_runtime))}")


def run_profiling(
    grpc_request_func: Callable,
    experiment_name: str,
    setup: str,
    code: str,
    trials: int,
    repetitions: int,
):
    """Print percentile statistics for given profiling experiment.

    :param Callable grpc_request_func: Function to execute grpc request
    :param str experiment_name: Experiment name
    :param str setup: Setup code to run
    :param str code: Code to profile
    :param int trials: Number of trials
    :param int repetitions: Number of repetitions to average over per trial
    """
    print()

    # Warm up
    get_measurement(
        grpc_request_func=grpc_request_func,
        setup=setup,
        code=code,
        repetitions=1
    )

    # Run actual experiment
    timings = []
    for i in range(trials):
        print(
            f"{POSIX_LINE_CLEAR}Running {experiment_name}, "
            "trial {i+1}/{trials}...",
            end="\r",
        )
        try:
            timings.append(
                get_measurement(
                    grpc_request_func=grpc_request_func,
                    setup=setup,
                    code=code,
                    repetitions=repetitions,
                )
            )
        except Exception as ex:
            print()  # Persist the trial counter
            raise ex

    print(f"{POSIX_LINE_CLEAR}{experiment_name} timings:")
    print_stats(timings)


def print_stats(timings: List[float]):
    df = pd.DataFrame({"timings": timings})

    indent = " " * 4
    quantiles = [0.25, 0.50, 0.75, 0.90, 0.99]
    print(f"{indent}Min:   {df.timings.min(): 0.2f} ms")
    for quantile in quantiles:
        print(
            f"{indent}{quantile * 100:0.1f}%: "
            "{df.timings.quantile(quantile): 0.2f} ms"
        )
    print(f"{indent}Max:   {df.timings.max(): 0.2f} ms")
    print(f"{indent}Mean:  {df.timings.mean(): 0.2f} ms")
