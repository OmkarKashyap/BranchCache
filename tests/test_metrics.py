from serving.metrics import get_metric_value, parse_prometheus_text

SAMPLE = """
# HELP vllm_prompt_tokens_total counter
# TYPE vllm_prompt_tokens_total counter
vllm_prompt_tokens_total{model="qwen"} 1234
vllm_time_to_first_token_seconds_sum{model="qwen"} 5.5
vllm_time_to_first_token_seconds_count{model="qwen"} 10
scrape_duration_seconds 0.002
"""


def test_parse_prometheus_text_reads_labeled_samples():
    metrics = parse_prometheus_text(SAMPLE)
    assert metrics["vllm_prompt_tokens_total"] == [({"model": "qwen"}, 1234.0)]


def test_parse_prometheus_text_reads_unlabeled_samples():
    metrics = parse_prometheus_text(SAMPLE)
    assert metrics["scrape_duration_seconds"] == [({}, 0.002)]


def test_get_metric_value_filters_by_label():
    metrics = parse_prometheus_text(SAMPLE)
    assert get_metric_value(metrics, "vllm_prompt_tokens_total", labels={"model": "qwen"}) == 1234.0


def test_get_metric_value_missing_metric_returns_none():
    metrics = parse_prometheus_text(SAMPLE)
    assert get_metric_value(metrics, "does_not_exist") is None
