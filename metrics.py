from typing import Iterable, List, Tuple

import numpy as np
from prometheus_client import REGISTRY, CollectorRegistry, Histogram, generate_latest
from prometheus_client.core import CounterMetricFamily, HistogramMetricFamily
from prometheus_client.parser import text_fd_to_metric_families

HISTOGRAM_PATH = "/artefact/train/histogram.prom"


class BaselineHistogramCollector(object):
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
        # Sample too small to determine, fallback to continuous histogram
        size = len(val)
        if size < 100:
            return False
        # Sample size too big, use 1% of max_samples to cap computation at 1ms
        if size > 1000:
            size = min(len(val), self.max_samples // 100)
            val = np.random.choice(val, size, replace=False)
        bins = np.unique(val)
        return len(bins) * 20 < size

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

            val = val[~np.isnan(val)]
            size_inf = size - len(val)

            # Clamp non-positive values to 0, assuming data is normalized
            sum_value = np.sum(val)
            val = val[val > 0]
            size_zero = size - size_inf - len(val)

            if self._is_discrete(val):
                bins, counts = np.unique(val, return_counts=True)
                counter = CounterMetricFamily(
                    name=f"feature_{i}_value_baseline",
                    documentation=f"Baseline values for feature: {name}",
                    labels=["bin"],
                )
                for j, b in enumerate(bins):
                    counter.add_metric(labels=[str(b)], value=counts[j])
                yield counter
                continue

            buckets = [["0.0", size_zero]]
            if len(val) == 0:
                buckets.append(["1.0", 0])
                buckets.append(["+Inf", size_inf])
            else:
                bins = self._get_bins(val)
                # Make all values negative to "le" as the bin upper bound
                count, _ = np.histogram(-val, bins=-np.flip([0] + bins))
                for j, c in enumerate(np.flip(count)):
                    buckets.append([str(bins[j]), buckets[-1][1] + c])
                buckets.append(["+Inf", buckets[-1][1] + size_inf])

            yield HistogramMetricFamily(
                name=f"feature_{i}_value_baseline",
                documentation=f"Baseline values for feature: {name}",
                buckets=buckets,
                sum_value=sum_value,
                labels=None,
                unit="",
            )


def log_histogram(data: Iterable[Tuple[str, List[float]]]):
    """Computes the histogram for each feature in a dataset.
    Saves to a local Prometheus file at HISTOGRAM_PATH.

    :param data: Iterable columns of name, data
    :type data: Iterable[Tuple[str, List[float]]]
    """
    registry = CollectorRegistry()
    collector = BaselineHistogramCollector(data)
    registry.register(collector)
    with open(HISTOGRAM_PATH, "wb") as f:
        f.write(generate_latest(registry=registry))


def serve_histogram(registry=REGISTRY) -> List[Histogram]:
    """Parses baseline histogram file to create serving time metrics.

    :param registry: The collector registry, defaults to REGISTRY
    :type registry: CollectorRegistry, optional
    :return: List of Prometheus Histgogram
    :rtype: List[Histogram]
    """
    hist = []
    with open(HISTOGRAM_PATH, "r") as f:
        for metric in text_fd_to_metric_families(f):
            if metric.type != "histogram":
                continue
            bins = tuple(
                sample.labels["le"]
                for sample in metric.samples
                if sample.name.endswith("_bucket")
            )
            hist.append(
                Histogram(
                    name=metric.name.replace("_baseline", ""),
                    documentation=metric.documentation.replace("Baseline", "Real time"),
                    buckets=bins,
                    registry=registry,
                )
            )
    return hist


if __name__ == "__main__":
    serve_histogram()
    with open("prom5", "wb") as f:
        f.write(generate_latest())
