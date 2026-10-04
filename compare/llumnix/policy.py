from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class RequestState:
    request_id: str
    service_class: str
    input_tokens: int
    predicted_output_tokens: int
    physical_tokens: int = 0
    queued: bool = True
    head_of_line: bool = False
    fake: bool = False

    @property
    def high_priority(self) -> bool:
        return self.service_class == "interactive"

    @property
    def demand_tokens(self) -> int:
        return self.input_tokens + self.predicted_output_tokens


@dataclass
class InstanceState:
    instance_id: str
    capacity_tokens: int
    requests: List[RequestState] = field(default_factory=list)
    active_requests: int = 0
    active: bool = True
    terminating: bool = False

    @property
    def batch_size(self) -> int:
        return max(1, self.active_requests)


@dataclass
class LlumnixPolicyConfig:
    priority_headroom_tokens: int = 1024
    migrate_out_freeness: float = 512.0
    migrate_in_freeness: float = 2048.0
    scale_up_freeness: float = 512.0
    scale_down_freeness: float = 4096.0
    min_instances: int = 2
    max_instances: int = 10


class LlumnixQueuePolicy:
    """Paper-faithful virtual-usage policy for the KServe queue mode.

    This mode moves only queued requests. Use the downloaded Llumnix backend
    for live migration of running requests and their KV cache.
    """

    def __init__(self, config: Optional[LlumnixPolicyConfig] = None):
        self.config = config or LlumnixPolicyConfig()

    def virtual_usage(self, request: RequestState, instance: InstanceState) -> float:
        if request.fake:
            return math.inf
        if request.queued:
            return float(request.demand_tokens if request.head_of_line else 0)
        headroom = 0.0
        if request.high_priority:
            high_count = max(1, sum(r.high_priority and not r.queued for r in instance.requests))
            headroom = self.config.priority_headroom_tokens / high_count
        return float(request.physical_tokens) + headroom

    def freeness(self, instance: InstanceState) -> float:
        if not instance.active:
            return -math.inf
        if instance.terminating:
            return -math.inf
        total_virtual = sum(self.virtual_usage(request, instance) for request in instance.requests)
        return (instance.capacity_tokens - total_virtual) / instance.batch_size

    def dispatch(self, instances: Iterable[InstanceState]) -> InstanceState:
        candidates = [item for item in instances if item.active and not item.terminating]
        if not candidates:
            raise RuntimeError("no active Llumnix instance is available")
        return max(candidates, key=lambda item: (self.freeness(item), item.instance_id))

    def migration_pairs(self, instances: Iterable[InstanceState]) -> List[Tuple[str, str]]:
        active = [item for item in instances if item.active]
        sources = sorted(
            [item for item in active if item.terminating or self.freeness(item) < self.config.migrate_out_freeness],
            key=lambda item: (self.freeness(item), item.instance_id),
        )
        destinations = sorted(
            [item for item in active if not item.terminating and self.freeness(item) > self.config.migrate_in_freeness],
            key=lambda item: (self.freeness(item), item.instance_id),
            reverse=True,
        )
        pairs = []
        used = set()
        for source, destination in zip(sources, destinations):
            if source.instance_id == destination.instance_id:
                continue
            if source.instance_id in used or destination.instance_id in used:
                continue
            pairs.append((source.instance_id, destination.instance_id))
            used.update((source.instance_id, destination.instance_id))
        return pairs

    @staticmethod
    def select_request_to_migrate(source: InstanceState, queued_only: bool = True) -> Optional[RequestState]:
        candidates = [request for request in source.requests if not queued_only or request.queued]
        if not candidates:
            return None
        # Paper: lower priority first, then shorter sequence to bound migration cost.
        return min(
            candidates,
            key=lambda request: (request.high_priority, request.demand_tokens, request.request_id),
        )

    def desired_instance_delta(self, instances: Iterable[InstanceState]) -> int:
        active = [item for item in instances if item.active and not item.terminating]
        if not active:
            return 1
        average = sum(self.freeness(item) for item in active) / len(active)
        if average < self.config.scale_up_freeness and len(active) < self.config.max_instances:
            return 1
        if average > self.config.scale_down_freeness and len(active) > self.config.min_instances:
            return -1
        return 0

    def choose_scale_down_victim(self, instances: Iterable[InstanceState]) -> Optional[InstanceState]:
        candidates = [item for item in instances if item.active and not item.terminating]
        if len(candidates) <= self.config.min_instances:
            return None
        return min(candidates, key=lambda item: (item.active_requests, len(item.requests), item.instance_id))

    @staticmethod
    def refresh_head_of_line(instance: InstanceState) -> None:
        queued = [request for request in instance.requests if request.queued]
        for request in queued:
            request.head_of_line = False
        if queued:
            # High scheduling priority first, FCFS is represented by list order.
            high = next((request for request in queued if request.high_priority), None)
            (high or queued[0]).head_of_line = True


def state_snapshot(instances: Iterable[InstanceState], policy: LlumnixQueuePolicy) -> Dict[str, dict]:
    result = {}
    for item in instances:
        freeness = policy.freeness(item)
        result[item.instance_id] = {
            "active": item.active,
            "terminating": item.terminating,
            "active_requests": item.active_requests,
            "queued_requests": sum(request.queued for request in item.requests),
            "freeness": freeness if math.isfinite(freeness) else None,
        }
    return result
