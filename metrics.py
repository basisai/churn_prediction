from abc import abstractmethod
from time import time
from typing import Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    Metric,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, HistogramMetricFamily
from prometheus_client.parser import text_fd_to_metric_families

HISTOGRAM_PATH = "/artefact/train/histogram.prom"


class FeatureMetric:
    """Custom Prometheus metric optimised for tracking feature and inference distribution during
    model training (baseline) and model serving.
    """

    BASELINE_NAME_PATTERN = "feature_{0}_value_baseline"
    BASELINE_DOC_PATTERN = "Baseline values for feature: {0}"

    def _get_serving_name_and_documentation_from_baseline(self, metric: Metric):
        return (
            metric.name.replace("_baseline", ""),
            metric.documentation.replace("Baseline", "Real time"),
        )

    @classmethod
    def load(cls, metric: Metric, registry: CollectorRegistry = REGISTRY):
        """Converts a Prometheus metric to feature metric and registers it for serving.

        :param metric: The dumped Prometheus metric.
        :type metric: Metric
        :param registry: The serving Prometheus registry, defaults to REGISTRY
        :type registry: CollectorRegistry, optional
        :return: A registered feature metric.
        :rtype: FeatureMetric
        """
        return cls(metric, registry)

    @abstractmethod
    def observe(self, value):
        raise NotImplementedError


class DiscreteFeature(FeatureMetric):
    """Handles discrete variables, including both numeric and non-numeric values.
    """

    BIN_LABEL = "bin"

    def __init__(self, metric: Metric, registry=REGISTRY):
        self.metric = Counter(
            *self._get_serving_name_and_documentation_from_baseline(metric),
            labelnames=(self.BIN_LABEL,),
            registry=registry,
        )
        for sample in metric.samples:
            self.metric.labels(**{self.BIN_LABEL: sample.labels[self.BIN_LABEL]}).inc(0)

    @classmethod
    def dump(cls, index: int, name: str, bin_to_count: Mapping[str, float]) -> Metric:
        """Converts a dictionary of bin to count to Prometheus counter.

        :param index: Index in the feature vector (must be the same for training and serving).
        :type index: int
        :param name: Name of the feature (used for documentation).
        :type name: str
        :param bin_to_count: Counts of items in each bin.
        :type bin_to_count: Mapping[str, float]
        :return: The converted Prometheus counter metric.
        :rtype: Metric
        """
        counter = CounterMetricFamily(
            name=cls.BASELINE_NAME_PATTERN.format(index),
            documentation=cls.BASELINE_DOC_PATTERN.format(name),
            labels=(cls.BIN_LABEL,),
        )
        for k, v in bin_to_count.items():
            counter.add_metric(labels=[k], value=v)
        return counter

    def observe(self, value):
        # Track None, NaN, Inf separately for discrete values
        self.metric.labels(bin=str(value)).inc()


class ContinuousFeature(FeatureMetric):
    """Handles continuous variables, including None, NaN, and Inf.
    """

    BIN_LABEL = "le"

    def __init__(self, metric: Metric, registry=REGISTRY):
        bins = tuple(
            sample.labels[self.BIN_LABEL]
            for sample in metric.samples
            if sample.name.endswith("_bucket")
        )
        self.metric = Histogram(
            *self._get_serving_name_and_documentation_from_baseline(metric),
            buckets=bins,
            registry=registry,
        )

    @classmethod
    def dump(
        cls,
        index: int,
        name: str,
        bin_to_count: Mapping[str, float],
        sum_value: Optional[float] = None,
    ) -> Metric:
        """Converts a dictionary of bin to count to Prometheus histogram.

        :param index: Index in the feature vector (must be the same for training and serving).
        :type index: int
        :param name: Name of the feature (used for documentation).
        :type name: str
        :param bin_to_count: Counts of items in each bin (must be inserted in ascending order of
            the bin's numerical value). The last bin can be "+Inf" to capture None, NaN, and inf.
        :type bin_to_count: Mapping[str, float]
        :param sum_value: The total value of all samples, defaults to raw bucket value * count
        :type sum_value: Optional[float], optional
        :return: The converted Prometheus histogram metric.
        :rtype: Metric
        """
        buckets = []
        accumulator = 0
        for k, v in bin_to_count.items():
            accumulator += v
            buckets.append([k, accumulator])

        if "+Inf" not in bin_to_count:
            buckets.append(["+Inf", buckets[-1][1]])

        return HistogramMetricFamily(
            name=cls.BASELINE_NAME_PATTERN.format(index),
            documentation=cls.BASELINE_DOC_PATTERN.format(name),
            buckets=buckets,
            sum_value=sum_value
            or sum(float(k) * v for k, v in bin_to_count.items() if k != "+Inf"),
        )

    def observe(self, value):
        if not value or value == float("inf") or value == float("nan"):
            self.metric._buckets[-1].inc(1)
        else:
            self.metric.observe(value)


class ComputedMetricCollector:
    def __init__(self, metric: List[Metric]):
        """A wrapper for manually computed baseline distribution of features metrics.

        :param metric: A list of Prometheus metrics returned from FeatureMetric.dump.
        :type metric: List[Metric]
        """
        self.metric = metric

    def collect(self):
        return self.metric


class FeatureMetricCollector:
    """Computes data distribution and exports as baseline feature metric.
    """

    SUPPORTED: Mapping[str, FeatureMetric] = {
        "histogram": ContinuousFeature,
        "counter": DiscreteFeature,
    }

    def __init__(self, data, max_samples: int = 100000):
        self.data: Iterable[Tuple[str, List[float]]] = data
        self.max_samples: int = max_samples

    def _is_discrete(self, val: List[float]) -> bool:
        """Litmus test to determine if val is discrete.

        :param val: Array of positive values
        :type val: List[float]
        :return: Whether input array only contains discrete values
        :rtype: bool
        """
        # Sample size too big, use 1% of max_samples to cap computation at 1ms
        size = len(val)
        if size > 1000:
            size = min(len(val), self.max_samples // 100)
            val = np.random.choice(val, size, replace=False)
        bins = np.unique(val)
        # Caps number of bins to 50
        return len(bins) < 3 or len(bins) * 20 < size

    @staticmethod
    def _get_bins(val: List[float]) -> List[float]:
        """Calculates the optimal bins for prometheus histogram.

        :param val: Array of positive values.
        :type val: List[float]
        :return: Upper bound of each bin (at least 2 bins)
        :rtype: List[float]
        """
        r_min = np.min(val)
        r_max = np.max(val)
        min_bins = 2
        max_bins = 50
        # Calculate bin width using either Freedman-Diaconis or Sturges estimator
        bin_edges = np.histogram_bin_edges(val, bins="auto")
        if len(bin_edges) < min_bins:
            return list(np.linspace(start=r_min, stop=r_max, num=min_bins))
        elif len(bin_edges) <= max_bins:
            return list(bin_edges)
        # Clamp to max_bins by estimating a good bin range to be more robust to outliers
        q75, q25 = np.percentile(val, [75, 25])
        iqr = q75 - q25
        width = 2 * iqr / max_bins
        start = max((q75 + q25) / 2 - iqr, r_min)
        stop = min(start + max_bins * width, r_max)
        # Take the minimum of range and 2x IQR to account for outliers
        edges = list(np.linspace(start=start, stop=stop, num=max_bins))
        prefix = [r_min] if start > r_min else []
        suffix = [r_max] if stop < r_max else []
        return prefix + edges + suffix

    def collect(self):
        """Calculates histogram bins using numpy and converts to Prometheus metric.

        :yield: The converted histogram metric for each feature.
        :rtype: HistogramMetricFamily
        """
        for i, col in enumerate(self.data):
            name, val = col
            # Assuming data is float. Categorical data should have been one-hot encoded
            # dtype=float will convert None values to np.nan as well
            val = np.asarray(val, dtype=float)
            size = len(val)
            # Sample without replacement to cap computation to about 3 seconds for 25 features
            if size > self.max_samples:
                size = self.max_samples
                val = np.random.choice(val, self.max_samples, replace=False)

            if self._is_discrete(val):
                bins, counts = np.unique(val, return_counts=True)
                bin_to_count = {str(bins[i]): counts[i] for i in range(len(bins))}
                yield DiscreteFeature.dump(
                    index=i, name=name, bin_to_count=bin_to_count,
                )
                continue

            val = val[~np.isnan(val)]
            val = val[~np.isinf(val)]
            size_inf = size - len(val)

            # Clamp non-positive values to 0, assuming data is normalized
            sum_value = np.sum(val)
            # val = val[val > 0]
            # size_zero = size - size_inf - len(val)

            if len(val) == 0:
                bin_to_count = {"0.0": 0}
            else:
                bins = self._get_bins(val)
                # Make all values negative to use "le" as the bin upper bound
                counts, _ = np.histogram(-val, bins=-np.flip([bins[0]] + bins))
                counts = np.flip(counts)
                bin_to_count = {str(bins[i]): counts[i] for i in range(len(bins))}

            bin_to_count["+Inf"] = size_inf
            yield ContinuousFeature.dump(
                index=i, name=name, bin_to_count=bin_to_count, sum_value=sum_value
            )


def log_raw(metric: List[Metric], path: str = HISTOGRAM_PATH):
    registry = CollectorRegistry()
    collector = ComputedMetricCollector(metric)
    registry.register(collector)
    with open(path, "wb") as f:
        f.write(generate_latest(registry=registry))


def log_histogram(data: Iterable[Tuple[str, List[float]]], path: str = HISTOGRAM_PATH):
    """Computes the histogram for each feature in a dataset.
    Saves to a local Prometheus file at HISTOGRAM_PATH.

    :param data: Iterable columns of name, data
    :type data: Iterable[Tuple[str, List[float]]]
    """
    registry = CollectorRegistry()
    collector = FeatureMetricCollector(data)
    registry.register(collector)
    with open(path, "wb") as f:
        f.write(generate_latest(registry=registry))


def serve_histogram(registry=REGISTRY) -> List[FeatureMetric]:
    """Parses baseline histogram file to create serving time metrics.

    :param registry: The collector registry, defaults to REGISTRY
    :type registry: CollectorRegistry, optional
    :return: List of custom Prometheus metrics
    :rtype: List[FeatureMetric]
    """
    features: List[FeatureMetric] = []
    with open(HISTOGRAM_PATH, "r") as f:
        for metric in text_fd_to_metric_families(f):
            # Ignore non-baseline metrics
            if not metric.name.endswith("_baseline"):
                continue
            # Ignore unsupported metric type
            if metric.type not in FeatureMetricCollector.SUPPORTED:
                continue
            serving = FeatureMetricCollector.SUPPORTED[metric.type].load(
                metric=metric, registry=registry
            )
            features.append(serving)
    return features


if __name__ == "__main__":
    serve_histogram()
    with open("prom5", "wb") as f:
        f.write(generate_latest())
