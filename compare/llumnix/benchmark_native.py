from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import time
import uuid
from pathlib import Path
from typing import List, Optional

import aiohttp

from compare.baseline_common.metrics import RequestMetric, summarize, write_jsonl
from compare.baseline_common.token_utils import count_tokens
from compare.baseline_common.workload import (
    WorkItem, build_per_class_workload, build_uniform_workload, load_prompts, load_window_counts,
)


async def send_llumnix_request(session, endpoint: str, item: WorkItem, arrival: float, sequence: int) -> RequestMetric:
    started = time.perf_counter()
    token_times = []
    final_text = ""
    status = None
    error = None
    payload = {
        "prompt": item.prompt,
        "stream": True,
        "max_tokens": item.max_tokens,
        "temperature": 0.7,
    }
    try:
        async with session.post(endpoint, json=payload, headers={"X-Llumnix-Trace": "true"}) as response:
            status = response.status
            if status >= 400:
                raise RuntimeError(f"HTTP {status}: {(await response.text())[:500]}")
            buffer = b""
            async for chunk in response.content.iter_any():
                buffer += chunk
                while b"\0" in buffer:
                    raw, buffer = buffer.split(b"\0", 1)
                    if not raw.strip():
                        continue
                    event = json.loads(raw.decode("utf-8"))
                    values = event.get("text") or []
                    if values:
                        candidate = str(values[0])
                        if len(candidate) > len(final_text):
                            final_text = candidate
                            token_times.append(time.perf_counter())
    except Exception as exc:
        error = str(exc)
    ended = time.perf_counter()
    generated = final_text[len(item.prompt):] if final_text.startswith(item.prompt) else final_text
    return RequestMetric(
        request_id=f"llumnix-native-{sequence}-{uuid.uuid4().hex[:8]}",
        service_class=item.service_class,
        endpoint=endpoint,
        input_tokens=item.input_tokens,
        output_tokens=count_tokens(generated),
        arrival_s=arrival,
        queue_ms=max(0.0, (started - arrival) * 1000.0),
        ttft_ms=(token_times[0] - started) * 1000.0 if token_times else None,
        itl_ms=[(right - left) * 1000.0 for left, right in zip(token_times, token_times[1:])],
        e2e_ms=(ended - started) * 1000.0,
        status=status,
        success=error is None and status is not None and 200 <= status < 300,
        error=error,
        scheduler={
            "baseline": "Llumnix-native",
            "live_kv_migration": True,
            "service_class_used_for_native_priority": False,
        },
    )


def resource_summary(
    duration_s: float,
    initial_instances: int,
    instance_csv: Optional[str],
    server_log: Optional[str],
) -> dict:
    average_instances = float(initial_instances)
    if instance_csv and Path(instance_csv).exists():
        rows = []
        with Path(instance_csv).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    rows.append((float(row["timestamp"]), float(row["num_instances"])))
                except (KeyError, TypeError, ValueError):
                    continue
        if len(rows) >= 2:
            rows.sort()
            weighted = sum((right[0] - left[0]) * left[1] for left, right in zip(rows, rows[1:]))
            span = rows[-1][0] - rows[0][0]
            if span > 0:
                average_instances = weighted / span

    migration_events = None
    if server_log and Path(server_log).exists():
        text = Path(server_log).read_text(encoding="utf-8", errors="replace")
        migration_events = len(re.findall(
            r"migrated request|migration completed|migrate_out.*success|begin migrate out",
            text,
            re.I,
        ))
    return {
        "average_active_instances": average_instances,
        "instance_seconds": average_instances * duration_s,
        "live_kv_migration_events_from_log": migration_events,
        "mode_disclosure": "real Llumnix running-request and KV-cache migration; fixed pool on current main branch",
    }


async def run(args, workload: List[WorkItem]) -> tuple[List[RequestMetric], float]:
    start = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def scheduled(item: WorkItem, sequence: int):
            target = start + item.offset_s
            if target > time.monotonic():
                await asyncio.sleep(target - time.monotonic())
            arrival = time.monotonic()
            return await send_llumnix_request(session, args.endpoint, item, arrival, sequence)

        records = await asyncio.gather(*[
            asyncio.create_task(scheduled(item, index)) for index, item in enumerate(workload)
        ])
    return records, time.monotonic() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the downloaded native Llumnix backend with common metrics.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234/generate")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--interactive-ratio", type=float, default=0.5)
    parser.add_argument("--workload-is-per-class", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--initial-instances", type=int, default=4)
    parser.add_argument("--instance-log-csv")
    parser.add_argument("--server-log")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    counts = load_window_counts(args.workload)
    if args.workload_is_per_class:
        workload = build_per_class_workload(prompts, counts, args.window_seconds, args.max_tokens)
    else:
        workload = build_uniform_workload(
            prompts, counts, args.window_seconds, args.interactive_ratio, args.max_tokens,
        )
    records, duration = asyncio.run(run(args, workload))
    resource = resource_summary(
        duration, args.initial_instances, args.instance_log_csv, args.server_log
    )
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
