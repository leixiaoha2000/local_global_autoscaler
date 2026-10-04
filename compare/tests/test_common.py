from compare.baseline_common.metrics import RequestMetric, summarize
from compare.baseline_common.workload import build_per_class_workload


def test_per_class_workload_replays_each_count_twice():
    workload = build_per_class_workload(["hello"], [2, 1], 2.0, 10)
    assert len(workload) == 6
    assert sum(item.service_class == "interactive" for item in workload) == 3
    assert sum(item.service_class == "batch" for item in workload) == 3


def test_summary_keeps_historical_tail_metrics_and_goodput():
    record = RequestMetric(
        request_id="r",
        service_class="interactive",
        endpoint="mock",
        input_tokens=10,
        output_tokens=5,
        arrival_s=0.0,
        queue_ms=1.0,
        ttft_ms=10.0,
        itl_ms=[2.0, 4.0],
        e2e_ms=20.0,
        status=200,
        success=True,
    )
    result = summarize(
        [record], 1.0,
        {"interactive": {"ttft_ms": 200.0, "itl_ms": 50.0}},
    )
    assert result["ttft_ms"]["p95"] == 10.0
    assert result["request_mean_itl_ms"]["p95"] == 3.0
    assert result["output_tps"] == 5.0
    assert result["goodput_rps"] == 1.0
