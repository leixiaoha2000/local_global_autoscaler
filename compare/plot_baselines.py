from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def nested(data: Dict[str, Any], *keys: str) -> Optional[float]:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def resolve_result(raw_path: str) -> Tuple[Path, Optional[Path]]:
    path = Path(raw_path)
    if path.is_dir():
        summary = path / "summary.json"
        requests = path / "requests.jsonl"
    else:
        summary = path
        requests = path.with_name("requests.jsonl")
    if not summary.exists():
        raise FileNotFoundError(f"summary not found: {summary}")
    return summary, requests if requests.exists() else None


def parse_specs(specs: Iterable[str]) -> List[Dict[str, Any]]:
    results = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"expected name=results/dir/or/summary.json, got: {spec}")
        name, raw_path = spec.split("=", 1)
        summary_path, requests_path = resolve_result(raw_path)
        results.append({
            "name": name,
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            "requests_path": requests_path,
        })
    return results


def finite(value: Optional[float]) -> float:
    return value if value is not None and math.isfinite(value) else 0.0


def plot_latency(results: List[Dict[str, Any]], output: Path, dpi: int) -> None:
    metrics = [
        ("TTFT", "ttft_ms"),
        ("ITL", "request_mean_itl_ms"),
        ("E2E", "e2e_ms"),
    ]
    percentiles = ("mean", "p95", "p99")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    width = 0.24
    x = list(range(len(results)))
    for axis, (title, key) in zip(axes, metrics):
        for offset, percentile in enumerate(percentiles):
            values = [finite(nested(item["summary"], key, percentile)) for item in results]
            positions = [value + (offset - 1) * width for value in x]
            axis.bar(positions, values, width=width, label=percentile.upper())
        axis.set_title(title)
        axis.set_ylabel("Latency (ms)")
        axis.set_xticks(x, [item["name"] for item in results], rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(output / "latency_percentiles.png", dpi=dpi)
    plt.close(fig)


def plot_system(results: List[Dict[str, Any]], output: Path, dpi: int) -> None:
    panels = [
        ("Output throughput", "tokens/s", lambda s: nested(s, "output_tps")),
        ("Total throughput", "tokens/s", lambda s: nested(s, "total_tps")),
        ("Goodput", "SLO requests/s", lambda s: nested(s, "goodput_rps")),
        ("SLO attainment", "%", lambda s: 100.0 * finite(nested(s, "slo_attainment"))),
        ("Average active instances", "instances", lambda s: nested(s, "resource", "average_active_instances")),
        ("Instance time", "instance-seconds", lambda s: nested(s, "resource", "instance_seconds")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    names = [item["name"] for item in results]
    for axis, (title, ylabel, accessor) in zip(axes.flat, panels):
        values = [finite(accessor(item["summary"])) for item in results]
        axis.bar(names, values)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "system_metrics.png", dpi=dpi)
    plt.close(fig)


def load_request_values(path: Path, key: str) -> List[float]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("success"):
                continue
            value = row.get(key)
            if value is not None:
                try:
                    numeric = float(value)
                    if math.isfinite(numeric):
                        values.append(numeric)
                except (TypeError, ValueError):
                    pass
    return sorted(values)


def plot_cdf(results: List[Dict[str, Any]], output: Path, dpi: int) -> bool:
    available = [item for item in results if item["requests_path"] is not None]
    if not available:
        return False
    metrics = [("TTFT", "ttft_ms"), ("ITL", "mean_itl_ms"), ("E2E", "e2e_ms")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    drew_any = False
    for axis, (title, key) in zip(axes, metrics):
        for item in available:
            values = load_request_values(item["requests_path"], key)
            if not values:
                continue
            probabilities = [(index + 1) / len(values) for index in range(len(values))]
            axis.plot(values, probabilities, label=item["name"], linewidth=2)
            drew_any = True
        axis.set_title(f"{title} CDF")
        axis.set_xlabel("Latency (ms)")
        axis.set_ylabel("CDF")
        axis.set_ylim(0, 1.01)
        axis.grid(alpha=0.25)
    if drew_any:
        axes[0].legend(frameon=False)
        fig.savefig(output / "request_cdf.png", dpi=dpi)
    plt.close(fig)
    return drew_any


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot comparable baseline summaries and request CDFs.")
    parser.add_argument("results", nargs="+", help="name=results_dir or name=summary.json")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = parse_specs(args.results)
    plot_latency(results, output, args.dpi)
    plot_system(results, output, args.dpi)
    cdf_written = plot_cdf(results, output, args.dpi)
    print(f"wrote {output / 'latency_percentiles.png'}")
    print(f"wrote {output / 'system_metrics.png'}")
    if cdf_written:
        print(f"wrote {output / 'request_cdf.png'}")


if __name__ == "__main__":
    main()
