"""HPA-compatible load generator for the external Llumnix controller."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import redis

from compare.baseline_common.metrics import summarize, write_jsonl
from compare.baseline_common.workload import (
    build_per_class_workload,
    build_uniform_workload,
    load_prompts,
    load_window_counts,
)
from compare.llumnix.benchmark_queue import LlumnixQueueBenchmark
from compare.llumnix.policy import LlumnixPolicyConfig, LlumnixQueuePolicy


def default_prompts_path() -> str:
    configured = os.getenv("PROMPTS_FILE")
    if configured:
        return configured
    primary = ROOT / "sharegpt_prompts.json"
    fallback = ROOT / "prompt.json"
    try:
        with primary.open("r", encoding="utf-8") as handle:
            json.load(handle)
        return str(primary)
    except (OSError, json.JSONDecodeError):
        if fallback.exists():
            print(
                f"[Llumnix load] warning: {primary} is unavailable or invalid; using {fallback}",
                file=sys.stderr,
                flush=True,
            )
            return str(fallback)
        return str(primary)


def wait_until_ready(store: redis.Redis, instances_path: Path) -> None:
    while not store.get("llumnix:controller_ready"):
        print("waiting for python ttft.py (llumnix:controller_ready)", flush=True)
        time.sleep(1)
    while not instances_path.exists():
        print(f"waiting for runtime endpoint file: {instances_path}", flush=True)
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay HPA-style workload through the external Llumnix queue controller."
    )
    parser.add_argument(
        "--instances",
        default=os.getenv(
            "LLUMNIX_INSTANCES_FILE", str(Path(__file__).with_name("instances.runtime.json"))
        ),
    )
    parser.add_argument("--prompts", default=default_prompts_path())
    # parser.add_argument(
    #     "--workload", default=os.getenv("WORKLOAD_FILE", str(ROOT / "shiyong" / "calary_sampled.csv"))
    # )
    # parser.add_argument(
    #     "--workload", default=os.getenv("WORKLOAD_FILE", str(ROOT / "shiyong" / "calary2_sampled.csv"))
    # )
    # parser.add_argument(
    #     "--workload", default=os.getenv("WORKLOAD_FILE", str(ROOT / "shiyong" / "dianshang_sampled.csv"))
    # )
    parser.add_argument(
        "--workload", default=os.getenv("WORKLOAD_FILE", str(ROOT / "shiyong" / "dianshang2_sampled.csv"))
    )
    parser.add_argument(
        "--output-dir", default=os.getenv("OUTPUT_DIR", str(ROOT / "results" / "llumnix-queue"))
    )
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen-0.5B"))
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "127.0.0.1"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--window-seconds", type=float, default=float(os.getenv("WINDOW_SECONDS", "2")))
    parser.add_argument(
        "--request-scale",
        type=float,
        default=float(os.getenv("REQUEST_SCALE", "1.0")),
        help="Multiply every CSV window count by this value (1.0 preserves the HPA workload).",
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "100")))
    parser.add_argument("--min-instances", type=int, default=int(os.getenv("MIN_INSTANCES", "2")))
    parser.add_argument("--max-instances", type=int, default=int(os.getenv("POOL_SIZE", "10")))
    parser.add_argument(
        "--instance-capacity-tokens",
        type=int,
        default=int(os.getenv("INSTANCE_CAPACITY_TOKENS", "8192")),
    )
    parser.add_argument(
        "--priority-headroom-tokens",
        type=int,
        default=int(os.getenv("PRIORITY_HEADROOM_TOKENS", "1024")),
    )
    parser.add_argument(
        "--migrate-out-freeness", type=float, default=float(os.getenv("MIGRATE_OUT_FREENESS", "512"))
    )
    parser.add_argument(
        "--migrate-in-freeness", type=float, default=float(os.getenv("MIGRATE_IN_FREENESS", "2048"))
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.getenv("MAX_CONCURRENCY", "128")),
        help="Per-instance client concurrency; defaults to vLLM --max-num-seqs=128.",
    )
    parser.add_argument(
        "--scheduler-interval", type=float, default=float(os.getenv("SCHEDULER_INTERVAL", "0.05"))
    )
    parser.add_argument(
        "--migration-cooldown", type=float, default=float(os.getenv("MIGRATION_COOLDOWN", "0.5"))
    )
    parser.add_argument("--timeout", type=float, default=float(os.getenv("REQUEST_TIMEOUT", "300")))
    parser.add_argument(
        "--service-mode",
        choices=("batch", "interactive", "both"),
        default=os.getenv("SERVICE_MODE", "batch"),
        help="batch matches HPA load-batch-calary.py; both replays each CSV count per class",
    )
    args = parser.parse_args()

    if args.request_scale <= 0:
        parser.error("--request-scale must be greater than zero")
    if not 1 <= args.min_instances <= args.max_instances:
        parser.error("instance limits must satisfy 1 <= min <= max")

    store = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    store.ping()
    instances_path = Path(args.instances)
    wait_until_ready(store, instances_path)
    endpoints = json.loads(instances_path.read_text(encoding="utf-8"))
    if len(endpoints) < args.min_instances:
        raise SystemExit(
            f"runtime endpoint file contains {len(endpoints)} instances; need at least {args.min_instances}"
        )

    config = LlumnixPolicyConfig(
        priority_headroom_tokens=args.priority_headroom_tokens,
        migrate_out_freeness=args.migrate_out_freeness,
        migrate_in_freeness=args.migrate_in_freeness,
        min_instances=args.min_instances,
        max_instances=min(args.max_instances, len(endpoints)),
    )
    runner = LlumnixQueueBenchmark(
        endpoints,
        LlumnixQueuePolicy(config),
        args.model,
        args.max_concurrency,
        args.scheduler_interval,
        1.0,
        args.timeout,
        args.migration_cooldown,
        external_store=store,
    )
    for instance in runner.instances.values():
        instance.capacity_tokens = args.instance_capacity_tokens

    prompts = load_prompts(args.prompts)
    original_counts = load_window_counts(args.workload)
    counts = [max(0, int(round(count * args.request_scale))) for count in original_counts]
    print(
        "[Llumnix load] "
        f"mode={args.service_mode} endpoints={len(endpoints)} windows={len(counts)} "
        f"window_seconds={args.window_seconds:g} request_scale={args.request_scale:g} "
        f"count_range={min(counts)}..{max(counts)} total_requests={sum(counts)} "
        f"max_concurrency_per_instance={args.max_concurrency}",
        flush=True,
    )
    if args.service_mode == "both":
        workload = build_per_class_workload(prompts, counts, args.window_seconds, args.max_tokens)
    else:
        workload = build_uniform_workload(
            prompts,
            counts,
            args.window_seconds,
            1.0 if args.service_mode == "interactive" else 0.0,
            args.max_tokens,
        )

    records, resource, duration = asyncio.run(runner.run(workload))
    raw_controller_resource = store.get("llumnix:resource")
    if raw_controller_resource:
        resource.update(json.loads(raw_controller_resource))
    resource["queue_migrations"] = sum(
        int(record.scheduler.get("queue_migration_count", 0)) for record in records
    )
    resource["live_kv_migrations"] = 0

    slo = {
        "interactive": {"ttft_ms": 200.0, "itl_ms": 50.0},
        "batch": {"ttft_ms": 2000.0, "itl_ms": 100.0},
    }
    summary = summarize(records, duration, slo, resource)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "requests.jsonl", records)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
