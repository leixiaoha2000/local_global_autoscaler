from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from compare.baseline_common.metrics import percentile
from compare.baseline_common.token_utils import length_bucket


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TokenScale velocity profile from saturation-run JSONL.")
    parser.add_argument(
        "jsonl",
        nargs="+",
        help="One or more per-request JSONL files produced by fixed_pool_benchmark.py",
    )
    parser.add_argument("--output", default="profile.json")
    parser.add_argument("--ttft-slo-ms", type=float, default=200.0)
    parser.add_argument("--itl-slo-ms", type=float, default=50.0)
    parser.add_argument("--convertible-ratio", type=float, default=0.25)
    args = parser.parse_args()

    records = []
    for source_index, source in enumerate(args.jsonl):
        with Path(source).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["_profile_source_index"] = source_index
                    records.append(row)
    good = [
        row for row in records
        if row.get("success") and row.get("ttft_ms") is not None
        and row["ttft_ms"] <= args.ttft_slo_ms
        and (row.get("mean_itl_ms") is None or row["mean_itl_ms"] <= args.itl_slo_ms)
    ]
    if not good:
        raise SystemExit("no SLO-attaining requests found; run a lower-rate profiling sweep first")

    interactive_good = [row for row in good if row.get("service_class") == "interactive"] or good
    batch_good = [row for row in good if row.get("service_class") == "batch"] or good
    per_second_input = defaultdict(float)
    per_second_decode = defaultdict(lambda: defaultdict(float))
    for row in interactive_good:
        second = (row["_profile_source_index"], int(float(row["arrival_s"])))
        per_second_input[second] += int(row.get("input_tokens", 0))
    for row in batch_good:
        second = (row["_profile_source_index"], int(float(row["arrival_s"])))
        predicted_output = int(
            (row.get("scheduler") or {}).get("requested_max_tokens", row.get("output_tokens", 0))
        )
        bucket = length_bucket(int(row.get("input_tokens", 0)), predicted_output)
        per_second_decode[bucket][second] += int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0))

    prefill = percentile(per_second_input.values(), 95) or 1.0
    decode = {
        bucket: percentile(second_rates.values(), 95) or 1.0
        for bucket, second_rates in per_second_decode.items()
    }
    decode["default"] = min(decode.values())
    profile = {
        "profile_source": [str(Path(source).resolve()) for source in args.jsonl],
        "prefill_tokens_per_s": prefill,
        "convertible_prefill_tokens_per_s": prefill * args.convertible_ratio,
        "decode_tokens_per_s": decode,
        "method": "p95 of one-second SLO-attaining token-rate bins from a saturation sweep",
    }
    Path(args.output).write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
