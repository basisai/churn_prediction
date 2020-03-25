from typing import Iterable, List, Tuple

import numpy as np
from prometheus_client import REGISTRY, CollectorRegistry, Histogram, generate_latest
from prometheus_client.core import HistogramMetricFamily
from prometheus_client.parser import text_fd_to_metric_families

HISTOGRAM_PATH = "/artefact/train/histogram.prom"


class BaselineHistogramCollector(object):
    def __init__(self, data):
        self.data = data

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
        q75, q25 = np.percentile(val, [75, 25])
        if q75 == q25:
            # If iqr is 0, layout 50 bins at regular interval to cover the full range
            bins = min(int(np.sqrt(len(val))), 50)
            edges, width = np.linspace(start=r_min, stop=r_max, num=bins, retstep=True)
            return [r_min, r_max] if width == 0 else list(edges)
        # Calculate bin width using Freedman-Diaconis rule
        iqr = q75 - q25
        width = 2 * iqr / (len(val) ** (1 / 3))
        bins = int(np.ceil((r_max - r_min) / width))
        if bins <= 50:
            return list(np.linspace(start=r_min, stop=r_max, num=max(bins, 2)))
        # Clamp to 50 bins to reduce cardinality at serving time
        bins = 50
        width = 2 * iqr / bins
        start = max((q75 + q25) / 2 - iqr, r_min)
        stop = min(start + bins * width, r_max)
        # Take the minimum of range and 2x IQR
        edges = list(np.linspace(start=start, stop=stop, num=bins))
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
            val = np.asarray(val)
            size = len(val)
            # Sample 10% of the population to cap computation to about 3 seconds
            if size > 1000000:
                size = 100000
                val = np.random.choice(val, size)

            val = val[~np.isnan(val)]
            size_inf = size - len(val)
            val = val[val != np.array(None)]
            size_none = size - size_inf - len(val)

            # Clamp non-positive values to 0, assuming data is normalized
            sum_value = np.sum(val)
            val = val[val > 0]
            size_zero = size - size_inf - size_none - len(val)

            buckets = [["0.0", size_zero + size_none]]
            if len(val) == 0:
                buckets.append(["1.0", 0])
                buckets.append(["+Inf", size_inf])
            else:
                bins = self._get_bins(val)
                count, _ = np.histogram(val, bins=bins)
                if bins[0] != bins[1]:
                    # Adjust for lower bound since prometheus requires le
                    count_first = np.count_nonzero(val == bins[0])
                    buckets.append([
                        str(bins[0]),
                        buckets[-1][1] + count_first,
                    ])
                    count[0] -= count_first
                for j, c in enumerate(count):
                    buckets.append([
                        str(bins[j + 1]),
                        buckets[-1][1] + c,
                    ])
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
                sample.labels["le"] for sample in metric.samples if sample.name.endswith("_bucket")
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
