from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import aiohttp

from compare.baseline_common.client import send_streaming_completion
from compare.baseline_common.metrics import RequestMetric, summarize, write_jsonl
from compare.baseline_common.workload import (
    WorkItem, build_per_class_workload, build_uniform_workload, load_prompts, load_window_counts,
)
from compare.llumnix.policy import (
    InstanceState,
    LlumnixPolicyConfig,
    LlumnixQueuePolicy,
    RequestState,
    state_snapshot,
)


@dataclass
class PendingRequest:
    work: WorkItem
    sequence: int
    arrival: float
    request: RequestState
    instance_id: str
    future: asyncio.Future
    migrations: List[dict] = field(default_factory=list)
    last_migration_at: float = float("-inf")


class LlumnixQueueBenchmark:
    """KServe-compatible Llumnix adaptation with queued-request rescheduling."""

    def __init__(
        self,
        endpoint_config: List[dict],
        policy: LlumnixQueuePolicy,
        model: str,
        max_concurrency_per_instance: int,
        scheduler_interval_s: float,
        scaling_interval_s: float,
        timeout_s: float,
        migration_cooldown_s: float = 0.5,
        external_store=None,
    ):
        self.policy = policy
        self.model = model
        self.max_concurrency = max_concurrency_per_instance
        self.scheduler_interval_s = scheduler_interval_s
        self.scaling_interval_s = scaling_interval_s
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self.migration_cooldown_s = migration_cooldown_s
        self.external_store = external_store
        self.urls = {item["name"]: item["url"] for item in endpoint_config}
        self.instances: Dict[str, InstanceState] = {}
        for index, item in enumerate(endpoint_config):
            self.instances[item["name"]] = InstanceState(
                instance_id=item["name"],
                capacity_tokens=policy.config.priority_headroom_tokens * 8,
                active=index < policy.config.min_instances,
            )
        policy.config.max_instances = min(policy.config.max_instances, len(self.instances))
        self.pending: List[PendingRequest] = []
        self.records: List[RequestMetric] = []
        self.events: List[dict] = []
        self.running_tasks = set()
        self.stop = asyncio.Event()
        self.session = None
        self.start = 0.0
        self.last_resource_time = 0.0
        self.active_instance_seconds = 0.0
        self.last_scale_time = 0.0
        self.last_external_sync = float("-inf")
        self.external_sync_interval_s = 0.5

    def sync_external_controller(self) -> None:
        """Mirror Redis roles written by the host-side Llumnix controller."""
        if self.external_store is None:
            return
        pipe = self.external_store.pipeline()
        names = list(self.instances)
        for name in names:
            pipe.get(f"pod:{name}:role")
        roles = pipe.execute()
        for name, role in zip(names, roles):
            if role is None:
                continue
            instance = self.instances[name]
            instance.active = role in {"ROLE_LLUMNIX", "ROLE_DRAINING"}
            instance.terminating = role == "ROLE_DRAINING"

    def publish_scheduler_metrics(self) -> None:
        """Expose client-side virtual usage so the external scaler sees queued work."""
        if self.external_store is None:
            return
        now = time.time()
        pipe = self.external_store.pipeline()
        for name, instance in self.instances.items():
            if instance.active and not instance.terminating:
                freeness = self.policy.freeness(instance)
                if math.isfinite(freeness):
                    pipe.set(f"pod:{name}:llumnix_scheduler_freeness", freeness, ex=5)
            pipe.set(f"pod:{name}:llumnix_scheduler_running", instance.active_requests, ex=5)
            pipe.set(
                f"pod:{name}:llumnix_scheduler_pending",
                sum(request.queued for request in instance.requests),
                ex=5,
            )
            pipe.set(f"pod:{name}:llumnix_scheduler_heartbeat", now, ex=5)
        pipe.execute()

    def account_resources(self, now: float) -> None:
        if self.last_resource_time:
            active = sum(instance.active for instance in self.instances.values())
            self.active_instance_seconds += active * max(0.0, now - self.last_resource_time)
        self.last_resource_time = now

    async def submit(self, work: WorkItem, sequence: int) -> RequestMetric:
        target = self.start + work.offset_s
        if target > time.monotonic():
            await asyncio.sleep(target - time.monotonic())
        arrival = time.monotonic()
        request = RequestState(
            request_id=f"llumnix-q-{sequence}-{uuid.uuid4().hex[:8]}",
            service_class=work.service_class,
            input_tokens=work.input_tokens,
            predicted_output_tokens=work.max_tokens,
        )
        destination = self.policy.dispatch(self.instances.values())
        destination.requests.append(request)
        self.policy.refresh_head_of_line(destination)
        future = asyncio.get_running_loop().create_future()
        self.pending.append(PendingRequest(work, sequence, arrival, request, destination.instance_id, future))
        return await future

    def reschedule_queues(self, now: float) -> None:
        for instance in self.instances.values():
            self.policy.refresh_head_of_line(instance)
        for source_id, destination_id in self.policy.migration_pairs(self.instances.values()):
            source = self.instances[source_id]
            destination = self.instances[destination_id]
            request = self.policy.select_request_to_migrate(source, queued_only=True)
            if request is None:
                continue
            pending = next((item for item in self.pending if item.request is request), None)
            if pending is None:
                continue
            if now - pending.last_migration_at < self.migration_cooldown_s:
                continue
            source.requests.remove(request)
            destination.requests.append(request)
            pending.instance_id = destination_id
            pending.last_migration_at = now
            event = {"timestamp": now, "request_id": request.request_id, "from": source_id, "to": destination_id}
            pending.migrations.append(event)
            self.events.append({"type": "queue_migration", **event})
            self.policy.refresh_head_of_line(source)
            self.policy.refresh_head_of_line(destination)

    def scale(self, now: float) -> None:
        if now - self.last_scale_time < self.scaling_interval_s:
            return
        self.last_scale_time = now
        delta = self.policy.desired_instance_delta(self.instances.values())
        if delta > 0:
            candidate = next((item for item in self.instances.values() if not item.active), None)
            if candidate:
                candidate.active = True
                candidate.terminating = False
                self.events.append({"type": "scale_up", "timestamp": now, "instance": candidate.instance_id})
        elif delta < 0:
            victim = self.policy.choose_scale_down_victim(self.instances.values())
            if victim:
                victim.terminating = True
                self.events.append({"type": "drain_start", "timestamp": now, "instance": victim.instance_id})

        for instance in self.instances.values():
            if instance.terminating and instance.active_requests == 0 and not instance.requests:
                instance.active = False
                instance.terminating = False
                self.events.append({"type": "scale_down", "timestamp": now, "instance": instance.instance_id})

    def dispatch_ready(self) -> None:
        # High scheduling priority first, then FCFS within a class.
        for pending in sorted(self.pending, key=lambda item: (item.work.service_class != "interactive", item.arrival)):
            instance = self.instances[pending.instance_id]
            if not instance.active or instance.terminating:
                destination = self.policy.dispatch(self.instances.values())
                if pending.request in instance.requests:
                    instance.requests.remove(pending.request)
                destination.requests.append(pending.request)
                pending.instance_id = destination.instance_id
                instance = destination
            if instance.active_requests >= self.max_concurrency:
                continue
            self.pending.remove(pending)
            pending.request.queued = False
            pending.request.head_of_line = False
            pending.request.physical_tokens = pending.request.demand_tokens
            instance.active_requests += 1
            self.policy.refresh_head_of_line(instance)
            task = asyncio.create_task(self.execute(pending))
            self.running_tasks.add(task)
            task.add_done_callback(self.running_tasks.discard)

    async def execute(self, pending: PendingRequest) -> None:
        instance = self.instances[pending.instance_id]
        payload = {
            "model": self.model,
            "prompt": pending.work.prompt,
            "max_tokens": pending.work.max_tokens,
            "stream": True,
            "temperature": 0.7,
        }
        record = await send_streaming_completion(
            self.session,
            self.urls[instance.instance_id],
            payload,
            request_id=pending.request.request_id,
            service_class=pending.work.service_class,
            arrival_monotonic=pending.arrival,
            scheduler_metadata={
                "baseline": "Llumnix-queue",
                "instance": instance.instance_id,
                "queue_migration_count": len(pending.migrations),
                "queue_migrations": pending.migrations,
                "live_kv_migration": False,
            },
        )
        if pending.request in instance.requests:
            instance.requests.remove(pending.request)
        instance.active_requests = max(0, instance.active_requests - 1)
        self.policy.refresh_head_of_line(instance)
        self.records.append(record)
        if not pending.future.done():
            pending.future.set_result(record)

    async def scheduler_loop(self) -> None:
        while not self.stop.is_set() or self.pending or self.running_tasks:
            now = time.monotonic()
            sync_external = (
                self.external_store is not None
                and now - self.last_external_sync >= self.external_sync_interval_s
            )
            if sync_external:
                self.sync_external_controller()
            self.account_resources(now)
            self.reschedule_queues(now)
            if self.external_store is None:
                self.scale(now)
            self.dispatch_ready()
            if sync_external:
                self.publish_scheduler_metrics()
                self.last_external_sync = now
            await asyncio.sleep(self.scheduler_interval_s)

    async def run(self, workload: List[WorkItem]) -> tuple[List[RequestMetric], dict, float]:
        self.start = time.monotonic()
        self.last_resource_time = self.start
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=connector, timeout=self.timeout) as session:
            self.session = session
            scheduler = asyncio.create_task(self.scheduler_loop())
            records = await asyncio.gather(*[
                asyncio.create_task(self.submit(item, index)) for index, item in enumerate(workload)
            ])
            self.stop.set()
            await scheduler
        ended = time.monotonic()
        self.account_resources(ended)
        resource = {
            "instance_seconds": self.active_instance_seconds,
            "average_active_instances": self.active_instance_seconds / max(ended - self.start, 1e-9),
            "queue_migrations": sum(event["type"] == "queue_migration" for event in self.events),
            "live_kv_migrations": 0,
            "scale_up_events": sum(event["type"] == "scale_up" for event in self.events),
            "scale_down_events": sum(event["type"] == "scale_down" for event in self.events),
            "events": self.events,
            "final_state": state_snapshot(self.instances.values(), self.policy),
            "mode_disclosure": "queued requests only; running requests are not migrated",
        }
        return records, resource, ended - self.start


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Llumnix virtual-usage scheduling on existing KServe pods.")
    parser.add_argument("--instances", required=True, help="JSON list of {name,url} backend instances")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen-0.5B")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--interactive-ratio", type=float, default=0.5)
    parser.add_argument("--workload-is-per-class", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--min-instances", type=int, default=2)
    parser.add_argument("--max-instances", type=int, default=10)
    parser.add_argument("--instance-capacity-tokens", type=int, default=8192)
    parser.add_argument("--priority-headroom-tokens", type=int, default=1024)
    parser.add_argument("--migrate-out-freeness", type=float, default=512.0)
    parser.add_argument("--migrate-in-freeness", type=float, default=2048.0)
    parser.add_argument("--scale-up-freeness", type=float, default=512.0)
    parser.add_argument("--scale-down-freeness", type=float, default=4096.0)
    parser.add_argument("--max-concurrency", type=int, default=128)
    parser.add_argument("--scheduler-interval", type=float, default=0.05)
    parser.add_argument("--scaling-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--migration-cooldown", type=float, default=0.5)
    args = parser.parse_args()

    endpoints = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    config = LlumnixPolicyConfig(
        priority_headroom_tokens=args.priority_headroom_tokens,
        migrate_out_freeness=args.migrate_out_freeness,
        migrate_in_freeness=args.migrate_in_freeness,
        scale_up_freeness=args.scale_up_freeness,
        scale_down_freeness=args.scale_down_freeness,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
    )
    runner = LlumnixQueueBenchmark(
        endpoints,
        LlumnixQueuePolicy(config),
        args.model,
        args.max_concurrency,
        args.scheduler_interval,
        args.scaling_interval,
        args.timeout,
        args.migration_cooldown,
    )
    for instance in runner.instances.values():
        instance.capacity_tokens = args.instance_capacity_tokens
    prompts = load_prompts(args.prompts)
    counts = load_window_counts(args.workload)
    if args.workload_is_per_class:
        workload = build_per_class_workload(prompts, counts, args.window_seconds, args.max_tokens)
    else:
        workload = build_uniform_workload(
            prompts, counts, args.window_seconds, args.interactive_ratio, args.max_tokens,
        )
    records, resource, duration = asyncio.run(runner.run(workload))
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
