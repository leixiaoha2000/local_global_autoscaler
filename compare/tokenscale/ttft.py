"""HPA-compatible TokenScale global token-velocity controller."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import redis

from compare.baseline_common.kserve import load_kubernetes, wait_for_pool
from compare.tokenscale.policy import (
    Arrival,
    InstanceSnapshot,
    ROLE_WARM,
    TokenScaleConfig,
    TokenScalePolicy,
    VelocityProfile,
)


NAMESPACE = os.getenv("NAMESPACE", "like")
POOL_SIZE = int(os.getenv("POOL_SIZE", "10"))
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CONTROL_INTERVAL = float(os.getenv("CONTROL_INTERVAL_SECONDS", "0.5"))
PROFILE_PATH = Path(os.getenv("TOKENSCALE_PROFILE", str(Path(__file__).with_name("profile.json"))))
LOG_DIR = Path(os.getenv("LOG_DIR", str(ROOT / "logs")))


def make_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("TokenScaleController")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - [TokenScale] - %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "global-tokenscale.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


LOGGER = make_logger()


class ResourceTracker:
    def __init__(self, now: float):
        self.start = now
        self.last = now
        self.roles = {}
        self.role_seconds = {}
        self.switches = 0

    def update(self, now: float, roles: dict) -> dict:
        elapsed = max(0.0, now - self.last)
        for role in self.roles.values():
            self.role_seconds[role] = self.role_seconds.get(role, 0.0) + elapsed
        self.switches += sum(self.roles.get(name, ROLE_WARM) != role for name, role in roles.items())
        self.roles = dict(roles)
        self.last = now
        active_seconds = sum(value for role, value in self.role_seconds.items() if role != ROLE_WARM)
        duration = max(now - self.start, 1e-9)
        return {
            "instance_seconds": active_seconds,
            "average_active_instances": active_seconds / duration,
            "role_instance_seconds": self.role_seconds,
            "role_switches": self.switches,
            "mode_disclosure": "HPA-compatible TokenScale controller with Redis arrival stream",
        }


class Controller:
    def __init__(self):
        if POOL_SIZE < 3:
            raise SystemExit("TokenScale requires POOL_SIZE>=3 (interactive + batch + mixed)")
        if not PROFILE_PATH.exists():
            raise SystemExit(
                f"missing velocity profile: {PROFILE_PATH}; run offline_profiler.py first or set TOKENSCALE_PROFILE"
            )
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.redis.ping()
        profile = VelocityProfile.load(PROFILE_PATH)
        config = TokenScaleConfig(
            max_instances=POOL_SIZE,
            convertible_instances=int(os.getenv("CONVERTIBLE_INSTANCES", "1")),
            arrival_window_s=float(os.getenv("ARRIVAL_WINDOW_SECONDS", "2")),
            safety_factor=float(os.getenv("TOKENSCALE_SAFETY_FACTOR", "1.15")),
        )
        self.policy = TokenScalePolicy(profile, config)
        self.api = load_kubernetes()
        self.snapshots = {
            f"qwen-instance-{index:02d}": InstanceSnapshot(f"qwen-instance-{index:02d}")
            for index in range(1, POOL_SIZE + 1)
        }
        self.last_stream_id = "0-0"
        self.tracker = ResourceTracker(time.time())

    def initialize(self) -> None:
        LOGGER.info("waiting for %d physical KServe instances", POOL_SIZE)
        wait_for_pool(self.api, POOL_SIZE, NAMESPACE)
        keys = list(self.redis.scan_iter("tokenscale:*"))
        if keys:
            self.redis.delete(*keys)
        for name in self.snapshots:
            self.redis.delete(f"pod:{name}:role")
        self.apply_plan()
        LOGGER.info("controller ready; profile=%s", PROFILE_PATH)

    def consume_arrivals(self) -> int:
        entries = self.redis.xread({"tokenscale:arrivals": self.last_stream_id}, count=5000, block=1)
        count = 0
        for _, messages in entries:
            for stream_id, fields in messages:
                self.last_stream_id = stream_id
                try:
                    self.policy.observe(Arrival(
                        timestamp=float(fields["timestamp"]),
                        service_class=fields["service_class"],
                        input_tokens=int(fields["input_tokens"]),
                        predicted_output_tokens=int(fields["predicted_output_tokens"]),
                    ))
                    count += 1
                except (KeyError, TypeError, ValueError):
                    LOGGER.warning("ignored malformed arrival stream item %s", stream_id)
        return count

    def refresh_instance_metrics(self) -> None:
        pipe = self.redis.pipeline()
        for name in self.snapshots:
            pipe.get(f"pod:{name}:running_requests")
            pipe.get(f"pod:{name}:role")
        values = pipe.execute()
        for index, snapshot in enumerate(self.snapshots.values()):
            running, role = values[index * 2:index * 2 + 2]
            snapshot.active_requests = int(float(running or 0))
            snapshot.role = role or snapshot.role

    def apply_plan(self) -> None:
        now = time.time()
        self.refresh_instance_metrics()
        plan = self.policy.plan(now)
        roles = self.policy.assign_roles(self.snapshots.values(), plan)
        for name, role in roles.items():
            self.snapshots[name].role = role
        plan_data = {
            "timestamp": now,
            "interactive": plan.interactive,
            "batch": plan.batch,
            "mixed": plan.mixed,
            "warm": plan.warm,
            "input_tokens_per_s": plan.input_tokens_per_s,
            "predicted_tokens_per_s": plan.predicted_tokens_per_s,
            "bucket_rates": plan.bucket_rates,
            "burst": plan.burst,
            "overload": plan.overload,
        }
        resource = self.tracker.update(now, roles)
        pipe = self.redis.pipeline()
        for name, role in roles.items():
            pipe.set(f"pod:{name}:role", role)
        pipe.set("tokenscale:burst", int(plan.burst), ex=5)
        pipe.set("tokenscale:last_plan", json.dumps(plan_data, ensure_ascii=False), ex=5)
        pipe.set("tokenscale:resource", json.dumps(resource, ensure_ascii=False), ex=10)
        pipe.set("tokenscale:controller_ready", now, ex=5)
        pipe.execute()
        LOGGER.info(
            "plan I=%d B=%d M=%d W=%d input=%.1f tok/s predicted=%.1f tok/s burst=%s overload=%s",
            plan.interactive, plan.batch, plan.mixed, plan.warm,
            plan.input_tokens_per_s, plan.predicted_tokens_per_s, plan.burst, plan.overload,
        )

    def run(self) -> None:
        self.initialize()
        while True:
            started = time.time()
            try:
                self.consume_arrivals()
                self.apply_plan()
            except Exception:
                LOGGER.exception("controller loop failed")
            time.sleep(max(0.0, CONTROL_INTERVAL - (time.time() - started)))


if __name__ == "__main__":
    Controller().run()

