import requests


def fetch_raw_metrics(base_url, timeout=5):
    resp = requests.get(f"{base_url.rstrip('/')}/metrics", timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_prometheus_text(text):
    """Turns Prometheus exposition text into {name: [(labels, value), ...]}.

    Handles the plain-text format vLLM/SGLang both expose on /metrics. Not a
    full spec-compliant parser (doesn't worry about commas inside quoted
    label values, for instance) but that's never come up in either engine's
    output so far.
    """
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "{" in line:
            name, rest = line.split("{", 1)
            label_str, value_str = rest.rsplit("}", 1)
            labels = _parse_labels(label_str)
        else:
            name, value_str = line.rsplit(" ", 1)
            labels = {}

        metrics.setdefault(name.strip(), []).append((labels, float(value_str.strip())))

    return metrics


def _parse_labels(label_str):
    labels = {}
    for part in label_str.split(","):
        part = part.strip()
        if not part:
            continue
        key, value = part.split("=", 1)
        labels[key.strip()] = value.strip().strip('"')
    return labels


def get_metric_value(metrics, name, labels=None):
    samples = metrics.get(name, [])
    total = 0.0
    matched = False
    for sample_labels, value in samples:
        if labels and not all(sample_labels.get(k) == v for k, v in labels.items()):
            continue
        total += value
        matched = True
    return total if matched else None


def fetch_metrics(base_url, timeout=5):
    return parse_prometheus_text(fetch_raw_metrics(base_url, timeout=timeout))
