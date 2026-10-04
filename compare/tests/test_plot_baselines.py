from __future__ import annotations

import json

from compare.plot_baselines import parse_specs, plot_cdf, plot_latency, plot_system


def test_summary_and_request_plots(tmp_path):
    result_dir = tmp_path / "baseline"
    result_dir.mkdir()
    summary = {
        "ttft_ms": {"mean": 10, "p95": 20, "p99": 30},
        "request_mean_itl_ms": {"mean": 2, "p95": 3, "p99": 4},
        "e2e_ms": {"mean": 100, "p95": 150, "p99": 180},
        "output_tps": 50,
        "total_tps": 75,
        "goodput_rps": 2,
        "slo_attainment": 0.9,
        "resource": {"average_active_instances": 2, "instance_seconds": 20},
    }
    (result_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    request = {
        "success": True,
        "ttft_ms": 10,
        "mean_itl_ms": 2,
        "e2e_ms": 100,
    }
    (result_dir / "requests.jsonl").write_text(json.dumps(request) + "\n", encoding="utf-8")

    results = parse_specs([f"test={result_dir}"])
    output = tmp_path / "figures"
    output.mkdir()
    plot_latency(results, output, 72)
    plot_system(results, output, 72)
    assert plot_cdf(results, output, 72)

    assert (output / "latency_percentiles.png").stat().st_size > 0
    assert (output / "system_metrics.png").stat().st_size > 0
    assert (output / "request_cdf.png").stat().st_size > 0
