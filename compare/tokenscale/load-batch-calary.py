"""HPA-compatible load generator for the external TokenScale controller."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aiohttp
import redis

from compare.baseline_common.client import send_streaming_completion
from compare.baseline_common.kserve import discover_instances, load_kubernetes
from compare.baseline_common.metrics import summarize, write_jsonl
from compare.baseline_common.workload import (
    build_per_class_workload,
    build_uniform_workload,
    load_prompts,
    load_window_counts,
)
from compare.tokenscale.policy import (
    InstanceSnapshot,
    ROLE_BATCH,
    ROLE_INTERACTIVE,
    ROLE_MIXED,
    TokenScalePolicy,
    VelocityProfile,
)


class ExternalTokenScaleLoad:
    def __init__(self, args):
        self.args = args
        self.redis = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
        self.redis.ping()
        self.api = load_kubernetes()
        self.policy = TokenScalePolicy(VelocityProfile.load(args.profile))
        self.snapshots = {}
        self.urls = {}
        self.records = []
        self.stop = asyncio.Event()
        self.start = 0.0

    async def discovery_loop(self) -> None:
        while not self.stop.is_set():
            try:
                instances = discover_instances(self.api, self.args.namespace)
                pipe = self.redis.pipeline()
                names = sorted(instances)
                for name in names:
                    pipe.get(f"pod:{name}:role")
                roles = pipe.execute()
                for name, role in zip(names, roles):
                    snapshot = self.snapshots.setdefault(name, InstanceSnapshot(name))
                    snapshot.role = role or snapshot.role
                    self.urls[name] = f"http://{instances[name]}:{self.args.target_port}/v1/completions"
            except Exception as exc:
                print(f"[TokenScale load] discovery failed: {exc}", file=sys.stderr, flush=True)
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    async def wait_until_ready(self) -> None:
        while not self.redis.get("tokenscale:controller_ready"):
            print("waiting for python ttft.py (tokenscale:controller_ready)", flush=True)
            await asyncio.sleep(1)
        while len(self.urls) < 3:
            print(f"waiting for routable KServe endpoints ({len(self.urls)}/3)", flush=True)
            await asyncio.sleep(1)

    async def choose(self, item):
        eligible_roles = {ROLE_INTERACTIVE, ROLE_MIXED} if item.service_class == "interactive" else {ROLE_BATCH, ROLE_MIXED}
        while True:
            available = {
                name: snapshot for name, snapshot in self.snapshots.items()
                if name in self.urls and snapshot.role in eligible_roles
            }
            if available:
                burst = self.redis.get("tokenscale:burst") == "1"
                return self.policy.choose_instance(
                    item.service_class, item.input_tokens, item.max_tokens, available, burst,
                )
            await asyncio.sleep(0.05)

    async def run_one(self, session, item, sequence):
        target = self.start + item.offset_s
        if target > time.monotonic():
            await asyncio.sleep(target - time.monotonic())
        arrival = time.monotonic()
        self.redis.xadd("tokenscale:arrivals", {
            "timestamp": time.time(),
            "service_class": item.service_class,
            "input_tokens": item.input_tokens,
            "predicted_output_tokens": item.max_tokens,
        }, maxlen=50000, approximate=True)
        chosen = await self.choose(item)
        chosen.active_requests += 1
        chosen.inflight_input_tokens += item.input_tokens
        chosen.inflight_output_tokens += item.max_tokens
        used_mixed = item.service_class == "interactive" and chosen.role == ROLE_MIXED
        if used_mixed:
            chosen.mixed_prefills += 1
        try:
            return await send_streaming_completion(
                session,
                self.urls[chosen.instance_id],
                {
                    "model": self.args.model,
                    "prompt": item.prompt,
                    "max_tokens": item.max_tokens,
                    "stream": True,
                    "temperature": 0.7,
                },
                f"tokenscale-hpa-{sequence}-{uuid.uuid4().hex[:8]}",
                item.service_class,
                arrival,
                {
                    "baseline": "TokenScale-adapted-hpa-structure",
                    "instance": chosen.instance_id,
                    "role": chosen.role,
                    "used_convertible": used_mixed,
                },
            )
        finally:
            chosen.active_requests = max(0, chosen.active_requests - 1)
            chosen.inflight_input_tokens = max(0, chosen.inflight_input_tokens - item.input_tokens)
            chosen.inflight_output_tokens = max(0, chosen.inflight_output_tokens - item.max_tokens)
            if used_mixed:
                chosen.mixed_prefills = max(0, chosen.mixed_prefills - 1)

    async def run(self, workload):
        discovery = asyncio.create_task(self.discovery_loop())
        await self.wait_until_ready()
        self.start = time.monotonic()
        timeout = aiohttp.ClientTimeout(total=self.args.timeout)
        async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
            records = await asyncio.gather(*[
                asyncio.create_task(self.run_one(session, item, index))
                for index, item in enumerate(workload)
            ])
        duration = time.monotonic() - self.start
        self.stop.set()
        await discovery
        return records, duration


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay both HPA-style service classes through TokenScale.")
    parser.add_argument("--profile", default=os.getenv("TOKENSCALE_PROFILE", str(Path(__file__).with_name("profile.json"))))
    parser.add_argument("--prompts", default=os.getenv("PROMPTS_FILE", str(ROOT / "sharegpt_prompts.json")))
    parser.add_argument("--workload", default=os.getenv("WORKLOAD_FILE", str(ROOT / "shiyong" / "calary_sampled.csv")))
    # parser.add_argument("--workload", default=os.getenv("WORKLOAD_FILE", str(ROOT / "shiyong" / "calary_sampled.csv")))
    # parser.add_argument("--workload", default=os.getenv("WORKLOAD_FILE", str(ROOT / "shiyong" / "calary2_sampled.csv")))
    # parser.add_argument("--workload", default=os.getenv("WORKLOAD_FILE", str(ROOT / "shiyong" / "dianshang_sampled.csv")))
    # parser.add_argument("--workload", default=os.getenv("WORKLOAD_FILE", str(ROOT / "shiyong" / "dianshang2_sampled.csv")))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", str(ROOT / "results" / "tokenscale")))
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen-0.5B"))
    parser.add_argument("--namespace", default=os.getenv("NAMESPACE", "like"))
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "127.0.0.1"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--target-port", type=int, default=8081)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--service-mode",
        choices=("batch", "interactive", "both"),
        default=os.getenv("SERVICE_MODE", "batch"),
        help="batch matches HPA load-batch-calary.py; both replays the CSV once per class",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    counts = load_window_counts(args.workload)
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
    runner = ExternalTokenScaleLoad(args)
    records, duration = asyncio.run(runner.run(workload))
    raw_resource = runner.redis.get("tokenscale:resource")
    resource = json.loads(raw_resource) if raw_resource else {
        "mode_disclosure": "TokenScale external controller resource snapshot unavailable"
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
