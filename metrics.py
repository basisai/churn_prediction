import numpy as np
from prometheus_client import CollectorRegistry, Histogram, generate_latest
from prometheus_client.parser import text_fd_to_metric_families

HISTOGRAM_PATH = "/artefact/train/histogram.prom"

def _freedman_diaconis_bins(a):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    # From https://stats.stackexchange.com/questions/798/
    if len(a) < 2:
        return 1
    q75, q25 = np.percentile(a, [75 ,25])
    iqr = q75 - q25
    h = 2 * iqr / (len(a) ** (1 / 3))
    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        bins = int(np.sqrt(len(a)))
        return bins, (np.max(a) - np.min(a)) / bins
    else:
        return int(np.ceil((np.max(a) - np.min(a)) / h)), h

def log_histogram(data):
    """Computes the histogram for each feature in a dataset.

    :param data: Iterable columns of data
    :type data: Iterable[List[float]]
    """
    registry = CollectorRegistry()
    for i, col in enumerate(data):
        name, val = col
        val = np.asarray(val)
        bins, width = _freedman_diaconis_bins(val)
        bins = min(bins, 50)
        first = max(np.mean(val) - width * bins / 2, width)
        last = first + width * bins
        metric = Histogram(
            name=f"feature_{i}_value_baseline",
            documentation=f"Baseline values for feature: {name}",
            buckets=tuple([0] + list(np.linspace(start=first, stop=last, num=bins))),
            registry=registry,
        )
        for v in val:
            metric.observe(v)
    # push_to_gateway(gateway="prometheus-pushgateway.core.svc", job="run_step_id", registry=REGISTRY)
    with open(HISTOGRAM_PATH, "wb") as f:
        f.write(generate_latest(registry=registry))

def main():
    with open(HISTOGRAM_PATH, "r") as f:
        for metric in text_fd_to_metric_families(f):
            if metric.type != "histogram":
                continue
            Histogram(
                name=metric.name.replace("_baseline", ""),
                documentation=metric.documentation,
                buckets=tuple(sample.labels["le"] for sample in metric.samples if sample.name.endswith("_bucket")),
            )

    with open("prom5", "wb") as f:
        f.write(generate_latest())

if __name__ == "__main__":
    main()
