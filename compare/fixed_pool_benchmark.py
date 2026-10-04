from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import List

import aiohttp

from compare.baseline_common.client import send_streaming_completion
from compare.baseline_common.metrics import RequestMetric, summarize, write_jsonl
from compare.baseline_common.workload import (
    WorkItem, build_per_class_workload, build_uniform_workload, load_prompts, load_window_counts,
)


async def run(args, endpoints: List[dict], workload: List[WorkItem]) -> tuple[List[RequestMetric], float]:
    start = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def scheduled(item: WorkItem, sequence: int):
            target = start + item.offset_s
            if target > time.monotonic():
                await asyncio.sleep(target - time.monotonic())
            arrival = time.monotonic()
            endpoint = endpoints[sequence % len(endpoints)]
            return await send_streaming_completion(
                session,
                endpoint["url"],
                {
                    "model": args.model,
                    "prompt": item.prompt,
                    "max_tokens": item.max_tokens,
                    "stream": True,
                    "temperature": 0.7,
                },
                f"fixed-{sequence}-{uuid.uuid4().hex[:8]}",
                item.service_class,
                arrival,
                {
                    "baseline": "fixed-round-robin",
                    "instance": endpoint["name"],
                    # TokenScale profiles the predicted output demand. Keeping
                    # the requested cap avoids putting an early-EOS response in
                    # the wrong output-length bucket.
                    "requested_max_tokens": item.max_tokens,
                },
            )

        records = await asyncio.gather(*[
            asyncio.create_task(scheduled(item, index)) for index, item in enumerate(workload)
        ])
    return records, time.monotonic() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-pool round-robin benchmark and TokenScale profiler input.")
    parser.add_argument("--instances", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen-0.5B")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--interactive-ratio", type=float, default=0.5)
    parser.add_argument("--workload-is-per-class", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    endpoints = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    if not endpoints:
        raise SystemExit("instances file is empty")
    prompts = load_prompts(args.prompts)
    counts = load_window_counts(args.workload)
    if args.workload_is_per_class:
        workload = build_per_class_workload(prompts, counts, args.window_seconds, args.max_tokens)
    else:
        workload = build_uniform_workload(
            prompts, counts, args.window_seconds, args.interactive_ratio, args.max_tokens,
        )
    records, duration = asyncio.run(run(args, endpoints, workload))
    resource = {
        "average_active_instances": float(len(endpoints)),
        "instance_seconds": len(endpoints) * duration,
    }
    slo = {
        "interactive": {"ttft_ms": 200.0, "itl_ms": 50.0},
        "batch": {"ttft_ms": 2000.0, "itl_ms": 100.0},
    }
    summary = summarize(records, duration, slo, resource)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "requests.jsonl", records)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
