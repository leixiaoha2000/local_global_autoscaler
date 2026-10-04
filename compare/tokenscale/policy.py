from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Union

from compare.baseline_common.token_utils import length_bucket

ROLE_INTERACTIVE = "ROLE_INTERACTIVE"
ROLE_BATCH = "ROLE_BATCH"
ROLE_MIXED = "ROLE_MIXED"
ROLE_WARM = "ROLE_WARM"


@dataclass(frozen=True)
class Arrival:
    timestamp: float
    service_class: str
    input_tokens: int
    predicted_output_tokens: int

    @property
    def bucket(self) -> str:
        return length_bucket(self.input_tokens, self.predicted_output_tokens)


@dataclass
class InstanceSnapshot:
    instance_id: str
    role: str = ROLE_WARM
    inflight_input_tokens: int = 0
    inflight_output_tokens: int = 0
    active_requests: int = 0
    mixed_prefills: int = 0


@dataclass
class VelocityProfile:
    prefill_tokens_per_s: float
    decode_tokens_per_s: Dict[str, float]
    convertible_prefill_tokens_per_s: float
    profile_source: str = "manual"

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VelocityProfile":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            prefill_tokens_per_s=float(raw["prefill_tokens_per_s"]),
            decode_tokens_per_s={key: float(value) for key, value in raw["decode_tokens_per_s"].items()},
            convertible_prefill_tokens_per_s=float(raw["convertible_prefill_tokens_per_s"]),
            profile_source=str(raw.get("profile_source", path)),
        )

    def decode_velocity(self, bucket: str) -> float:
        if bucket in self.decode_tokens_per_s:
            return self.decode_tokens_per_s[bucket]
        if "default" in self.decode_tokens_per_s:
            return self.decode_tokens_per_s["default"]
        return min(self.decode_tokens_per_s.values())


@dataclass
class TokenScaleConfig:
    arrival_window_s: float = 2.0
    safety_factor: float = 1.15
    min_interactive: int = 1
    min_batch: int = 1
    convertible_instances: int = 1
    max_instances: int = 10
    burst_factor: float = 1.5
    burst_floor_tokens_s: float = 32.0
    ema_alpha: float = 0.3
    scale_down_hold_intervals: int = 3
    interactive_ttft_slo_ms: float = 200.0
    batch_ttft_slo_ms: float = 2000.0


@dataclass
class ScalePlan:
    interactive: int
    batch: int
    mixed: int
    warm: int
    input_tokens_per_s: float
    predicted_tokens_per_s: float
    burst: bool
    overload: bool
    bucket_rates: Dict[str, float] = field(default_factory=dict)

    @property
    def active(self) -> int:
        return self.interactive + self.batch + self.mixed


class TokenScalePolicy:
    """Token Velocity policy adapted from PD stages to two service classes.

    Interactive input-token velocity represents the TTFT-sensitive prefill
    pressure. Batch input+predicted-output velocity represents the longer-lived
    generation/KV-cache pressure. Mixed instances are the paper's Convertible
    Decoders in this repository's monolithic vLLM deployment.
    """

    def __init__(self, profile: VelocityProfile, config: Optional[TokenScaleConfig] = None):
        self.profile = profile
        self.config = config or TokenScaleConfig()
        self.arrivals: Deque[Arrival] = deque()
        self._ema_input_rate = 0.0
        self._last_plan: Optional[ScalePlan] = None
        self._down_streak = 0

    def observe(self, arrival: Arrival) -> None:
        self.arrivals.append(arrival)
        self._prune(arrival.timestamp)

    def _prune(self, now: float) -> None:
        cutoff = now - self.config.arrival_window_s
        while self.arrivals and self.arrivals[0].timestamp < cutoff:
            self.arrivals.popleft()

    def plan(self, now: Optional[float] = None) -> ScalePlan:
        now = time.monotonic() if now is None else now
        self._prune(now)
        window = max(self.config.arrival_window_s, 1e-9)
        interactive = [a for a in self.arrivals if a.service_class == "interactive"]
        batch = [a for a in self.arrivals if a.service_class != "interactive"]

        input_rate = sum(a.input_tokens for a in interactive) / window
        bucket_rates: Dict[str, float] = {}
        for arrival in batch:
            bucket_rates[arrival.bucket] = bucket_rates.get(arrival.bucket, 0.0) + (
                arrival.input_tokens + arrival.predicted_output_tokens
            ) / window
        predicted_rate = sum(bucket_rates.values())

        previous_ema = self._ema_input_rate
        burst = input_rate >= self.config.burst_floor_tokens_s and input_rate > max(
            self.config.burst_floor_tokens_s,
            previous_ema * self.config.burst_factor,
        )
        self._ema_input_rate = (
            self.config.ema_alpha * input_rate + (1.0 - self.config.ema_alpha) * previous_ema
        )

        required_interactive = max(
            self.config.min_interactive,
            math.ceil(self.config.safety_factor * input_rate / max(self.profile.prefill_tokens_per_s, 1e-9)),
        )
        fractional_batch = sum(
            rate / max(self.profile.decode_velocity(bucket), 1e-9)
            for bucket, rate in bucket_rates.items()
        )
        # Convertible instances normally contribute decode capacity, as in TokenScale.
        required_batch_total = max(
            self.config.min_batch + self.config.convertible_instances,
            math.ceil(self.config.safety_factor * fractional_batch),
        )
        required_batch = max(self.config.min_batch, required_batch_total - self.config.convertible_instances)
        mixed = self.config.convertible_instances

        requested = required_interactive + required_batch + mixed
        overload = requested > self.config.max_instances
        if overload:
            # Protect TTFT-sensitive interactive traffic and the fixed convertible buffer first.
            available_batch = self.config.max_instances - required_interactive - mixed
            required_batch = max(0, available_batch)
            if required_interactive + mixed > self.config.max_instances:
                required_interactive = max(0, self.config.max_instances - mixed)
                required_batch = 0

        candidate = ScalePlan(
            interactive=required_interactive,
            batch=required_batch,
            mixed=mixed,
            warm=max(0, self.config.max_instances - required_interactive - required_batch - mixed),
            input_tokens_per_s=input_rate,
            predicted_tokens_per_s=predicted_rate,
            burst=burst,
            overload=overload,
            bucket_rates=bucket_rates,
        )
        candidate = self._apply_scale_down_hysteresis(candidate)
        self._last_plan = candidate
        return candidate

    def _apply_scale_down_hysteresis(self, candidate: ScalePlan) -> ScalePlan:
        if self._last_plan is None or candidate.active >= self._last_plan.active:
            self._down_streak = 0
            return candidate
        self._down_streak += 1
        if self._down_streak < self.config.scale_down_hold_intervals:
            return ScalePlan(
                interactive=self._last_plan.interactive,
                batch=self._last_plan.batch,
                mixed=self._last_plan.mixed,
                warm=self._last_plan.warm,
                input_tokens_per_s=candidate.input_tokens_per_s,
                predicted_tokens_per_s=candidate.predicted_tokens_per_s,
                burst=candidate.burst,
                overload=candidate.overload,
                bucket_rates=candidate.bucket_rates,
            )
        self._down_streak = 0
        return candidate

    def assign_roles(
        self,
        instances: Iterable[InstanceSnapshot],
        plan: ScalePlan,
    ) -> Dict[str, str]:
        """Assign roles with minimal churn, keeping busy instances where possible."""
        snapshots = list(instances)
        if len(snapshots) < plan.active:
            raise ValueError(f"plan needs {plan.active} instances, only {len(snapshots)} supplied")
        assignments: Dict[str, str] = {}
        remaining = list(snapshots)

        def retain(role: str, count: int) -> None:
            candidates = sorted(
                (item for item in remaining if item.role == role),
                key=lambda item: (-item.active_requests, item.instance_id),
            )[:count]
            for item in candidates:
                assignments[item.instance_id] = role
                remaining.remove(item)

        retain(ROLE_MIXED, plan.mixed)
        retain(ROLE_INTERACTIVE, plan.interactive)
        retain(ROLE_BATCH, plan.batch)

        def fill(role: str, desired: int) -> None:
            missing = desired - sum(value == role for value in assignments.values())
            for item in sorted(remaining, key=lambda value: (value.active_requests, value.instance_id))[:missing]:
                assignments[item.instance_id] = role
                remaining.remove(item)

        fill(ROLE_MIXED, plan.mixed)
        fill(ROLE_INTERACTIVE, plan.interactive)
        fill(ROLE_BATCH, plan.batch)
        for item in remaining:
            assignments[item.instance_id] = ROLE_WARM
        return assignments

    def choose_instance(
        self,
        service_class: str,
        input_tokens: int,
        predicted_output_tokens: int,
        instances: Mapping[str, InstanceSnapshot],
        burst: bool,
    ) -> InstanceSnapshot:
        if not instances:
            raise RuntimeError("no instances are available")
        if service_class == "interactive":
            regular = [item for item in instances.values() if item.role == ROLE_INTERACTIVE]
            convertible = [
                item for item in instances.values()
                if item.role == ROLE_MIXED and item.mixed_prefills < 1
            ]
            def wait_ms(item: InstanceSnapshot, velocity: float) -> float:
                return (item.inflight_input_tokens + input_tokens) / max(velocity, 1e-9) * 1000.0

            regular.sort(key=lambda item: (wait_ms(item, self.profile.prefill_tokens_per_s), item.instance_id))
            if regular and wait_ms(regular[0], self.profile.prefill_tokens_per_s) <= self.config.interactive_ttft_slo_ms:
                return regular[0]
            convertible.sort(
                key=lambda item: (wait_ms(item, self.profile.convertible_prefill_tokens_per_s), item.instance_id)
            )
            if convertible and (burst or not regular):
                return convertible[0]
            if regular:
                return regular[0]
            if convertible:
                return convertible[0]
        else:
            bucket = length_bucket(input_tokens, predicted_output_tokens)
            velocity = self.profile.decode_velocity(bucket)
            candidates = [
                item for item in instances.values()
                if item.role in {ROLE_BATCH, ROLE_MIXED}
            ]
            if candidates:
                return min(
                    candidates,
                    key=lambda item: (
                        (item.inflight_input_tokens + item.inflight_output_tokens) / max(velocity, 1e-9),
                        item.instance_id,
                    ),
                )
        return min(instances.values(), key=lambda item: (item.active_requests, item.instance_id))
