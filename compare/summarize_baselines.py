from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def nested(data, *keys):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one comparable table from baseline summary.json files.")
    parser.add_argument("summaries", nargs="+", help="name=path/to/summary.json")
    parser.add_argument("--csv", default="baseline_summary.csv")
    args = parser.parse_args()
    fields = [
        "baseline", "requests", "success_rate", "slo_attainment", "goodput_rps",
        "ttft_mean_ms", "ttft_p95_ms", "ttft_p99_ms",
        "itl_mean_ms", "itl_p95_ms", "itl_p99_ms",
        "e2e_mean_ms", "e2e_p95_ms", "e2e_p99_ms",
        "input_tps", "output_tps", "total_tps",
        "average_active_instances", "instance_seconds", "migrations",
    ]
    rows = []
    for specification in args.summaries:
        if "=" not in specification:
            raise SystemExit(f"expected name=path, got: {specification}")
        name, raw_path = specification.split("=", 1)
        summary = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        resource = summary.get("resource", {})
        migrations = resource.get("live_kv_migration_events_from_log")
        if migrations is None:
            migrations = resource.get("queue_migrations")
        rows.append({
            "baseline": name,
            "requests": summary.get("requests"),
            "success_rate": summary.get("success_rate"),
            "slo_attainment": summary.get("slo_attainment"),
            "goodput_rps": summary.get("goodput_rps"),
            "ttft_mean_ms": nested(summary, "ttft_ms", "mean"),
            "ttft_p95_ms": nested(summary, "ttft_ms", "p95"),
            "ttft_p99_ms": nested(summary, "ttft_ms", "p99"),
            "itl_mean_ms": nested(summary, "request_mean_itl_ms", "mean"),
            "itl_p95_ms": nested(summary, "request_mean_itl_ms", "p95"),
            "itl_p99_ms": nested(summary, "request_mean_itl_ms", "p99"),
            "e2e_mean_ms": nested(summary, "e2e_ms", "mean"),
            "e2e_p95_ms": nested(summary, "e2e_ms", "p95"),
            "e2e_p99_ms": nested(summary, "e2e_ms", "p99"),
            "input_tps": summary.get("input_tps"),
            "output_tps": summary.get("output_tps"),
            "total_tps": summary.get("total_tps"),
            "average_active_instances": resource.get("average_active_instances"),
            "instance_seconds": resource.get("instance_seconds"),
            "migrations": migrations,
        })
    with Path(args.csv).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(" | ".join(fields))
    print(" | ".join(["---"] * len(fields)))
    for row in rows:
        print(" | ".join("" if row[field] is None else str(row[field]) for field in fields))


if __name__ == "__main__":
    main()
