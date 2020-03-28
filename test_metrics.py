from random import choices
from time import time

import numpy as np

from metrics import ContinuousFeature, DiscreteFeature, log_histogram, log_raw

SAMPLE_TRAINING_DATA = [
    ("first", [3.0, 3.0, 4.0, 5.0, 8.0, 10.0, 13.0, 13.0]),
    ("second", [13.0, 13.0, 14.0, 15.0, 18.0, 20.0, 23.0, np.nan]),
    ("third", [-1.0, 0, 2.0, 5.0, 8.0, 10.0, 13.0, 13.0, np.inf, np.NINF]),
    ("fourth", [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
]

SAMPLE_SERVING_DATA = np.array(
    [[73, 57], [68, 99], [25, 45], [33, 89], [50, 53], [79, 91], [94, 81], [3, 14]]
)

EXPECTED_HISTOGRAM_FILE = [
    "# HELP feature_0_value_baseline Baseline values for feature: first\n",
    "# TYPE feature_0_value_baseline histogram\n",
    'feature_0_value_baseline_bucket{le="3.0"} 2.0\n',
    'feature_0_value_baseline_bucket{le="5.5"} 4.0\n',
    'feature_0_value_baseline_bucket{le="8.0"} 5.0\n',
    'feature_0_value_baseline_bucket{le="10.5"} 6.0\n',
    'feature_0_value_baseline_bucket{le="13.0"} 8.0\n',
    'feature_0_value_baseline_bucket{le="+Inf"} 8.0\n',
    "feature_0_value_baseline_count 8.0\n",
    "feature_0_value_baseline_sum 59.0\n",
    "# HELP feature_1_value_baseline Baseline values for feature: second\n",
    "# TYPE feature_1_value_baseline histogram\n",
    'feature_1_value_baseline_bucket{le="13.0"} 2.0\n',
    'feature_1_value_baseline_bucket{le="15.5"} 4.0\n',
    'feature_1_value_baseline_bucket{le="18.0"} 5.0\n',
    'feature_1_value_baseline_bucket{le="20.5"} 6.0\n',
    'feature_1_value_baseline_bucket{le="23.0"} 7.0\n',
    'feature_1_value_baseline_bucket{le="+Inf"} 8.0\n',
    "feature_1_value_baseline_count 8.0\n",
    "feature_1_value_baseline_sum 116.0\n",
    "# HELP feature_2_value_baseline Baseline values for feature: third\n",
    "# TYPE feature_2_value_baseline histogram\n",
    'feature_2_value_baseline_bucket{le="-1.0"} 1.0\n',
    'feature_2_value_baseline_bucket{le="2.5"} 3.0\n',
    'feature_2_value_baseline_bucket{le="6.0"} 4.0\n',
    'feature_2_value_baseline_bucket{le="9.5"} 5.0\n',
    'feature_2_value_baseline_bucket{le="13.0"} 8.0\n',
    'feature_2_value_baseline_bucket{le="+Inf"} 10.0\n',
    "feature_2_value_baseline_count 10.0\n",
    "feature_2_value_baseline_sum 50.0\n",
    "# HELP feature_3_value_baseline_total Baseline values for feature: fourth\n",
    "# TYPE feature_3_value_baseline_total counter\n",
    'feature_3_value_baseline_total{bin="0.0"} 5.0\n',
    'feature_3_value_baseline_total{bin="1.0"} 3.0\n',
]


def test_collector():
    path = "./test_histogram.prom"
    log_histogram(SAMPLE_TRAINING_DATA, path=path)
    with open(path, "rb") as f:
        histogram_data = f.readlines()
        for i, line in enumerate(histogram_data):
            assert (
                EXPECTED_HISTOGRAM_FILE[i] == line.decode()
            ), f"Comparison failed at line {i + 1}"


def test_computed_collector():
    path = "./test_histogram.prom"
    log_raw(
        [
            ContinuousFeature.dump(
                index=0,
                name="first",
                bin_to_count={
                    "3.0": 2.0,
                    "5.5": 2.0,
                    "8.0": 1.0,
                    "10.5": 1.0,
                    "13.0": 2.0,
                },
                sum_value=59,
            ),
            ContinuousFeature.dump(
                index=1,
                name="second",
                bin_to_count={
                    "13.0": 2.0,
                    "15.5": 2.0,
                    "18.0": 1.0,
                    "20.5": 1.0,
                    "23.0": 1.0,
                    "+Inf": 1.0,
                },
                sum_value=116,
            ),
            ContinuousFeature.dump(
                index=2,
                name="third",
                bin_to_count={
                    "-1.0": 1.0,
                    "2.5": 2.0,
                    "6.0": 1.0,
                    "9.5": 1.0,
                    "13.0": 3.0,
                    "+Inf": 2.0,
                },
                sum_value=50,
            ),
            DiscreteFeature.dump(
                index=3, name="fourth", bin_to_count={"0.0": 5, "1.0": 3}
            ),
        ],
        path=path,
    )
    with open(path, "rb") as f:
        histogram_data = f.readlines()
        for i, line in enumerate(histogram_data):
            assert (
                EXPECTED_HISTOGRAM_FILE[i] == line.decode()
            ), f"Comparison failed at line {i + 1}"
