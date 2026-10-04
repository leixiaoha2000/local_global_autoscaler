"""HPA-compatible Llumnix global freeness controller."""

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


ROLE_ACTIVE = "ROLE_LLUMNIX"
ROLE_DRAINING = "ROLE_DRAINING"
ROLE_WARM = "ROLE_WARM"

NAMESPACE = os.getenv("NAMESPACE", "like")
POOL_SIZE = int(os.getenv("POOL_SIZE", "10"))
MIN_INSTANCES = int(os.getenv("MIN_INSTANCES", "2"))
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
TARGET_PORT = int(os.getenv("TARGET_PORT", "8081"))
CONTROL_INTERVAL = float(os.getenv("CONTROL_INTERVAL_SECONDS", "1"))
SCALE_COOLDOWN = float(os.getenv("SCALE_COOLDOWN_SECONDS", "2"))
SCALE_UP_FREENESS = float(os.getenv("SCALE_UP_FREENESS", "200"))
SCALE_DOWN_FREENESS = float(os.getenv("SCALE_DOWN_FREENESS", "500"))
SCALE_DOWN_HOLD = int(os.getenv("SCALE_DOWN_HOLD_INTERVALS", "3"))
PENDING_PER_INSTANCE_UP = int(os.getenv("PENDING_PER_INSTANCE_UP", "8"))
PENDING_HOLD = int(os.getenv("PENDING_HOLD_INTERVALS", "2"))
SIDECAR_HEARTBEAT_TTL = float(os.getenv("SIDECAR_HEARTBEAT_TTL_SECONDS", "15"))
RUNTIME_INSTANCES = Path(os.getenv(
    "LLUMNIX_INSTANCES_FILE", str(Path(__file__).with_name("instances.runtime.json"))
))
LOG_DIR = Path(os.getenv("LOG_DIR", str(ROOT / "logs")))


def make_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("LlumnixQueueController")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - [Llumnix-queue] - %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "global-llumnix.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


LOGGER = make_logger()


def write_instances(instances: dict) -> None:
    payload = [
        {"name": name, "url": f"http://{ip}:{TARGET_PORT}/v1/completions"}
        for name, ip in sorted(instances.items())
    ]
    temporary = RUNTIME_INSTANCES.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(RUNTIME_INSTANCES)


class ResourceTracker:
    def __init__(self, now: float):
        self.start = now
        self.last = now
        self.roles = {}
        self.instance_seconds = 0.0
        self.scale_up_events = 0
        self.scale_down_events = 0

    def update(self, now: float, roles: dict, action: str | None) -> dict:
        elapsed = max(0.0, now - self.last)
        self.instance_seconds += sum(
            role in {ROLE_ACTIVE, ROLE_DRAINING} for role in self.roles.values()
        ) * elapsed
        if action and action.startswith("scale_up"):
            self.scale_up_events += 1
        if action and action.startswith("scale_down_complete"):
            self.scale_down_events += 1
        self.roles = dict(roles)
        self.last = now
        duration = max(now - self.start, 1e-9)
        return {
            "instance_seconds": self.instance_seconds,
            "average_active_instances": self.instance_seconds / duration,
            "scale_up_events": self.scale_up_events,
            "scale_down_events": self.scale_down_events,
            "mode_disclosure": "Llumnix queue adaptation; queued requests migrate, running KV cache does not",
        }


class Controller:
    def __init__(self):
        if not 1 <= MIN_INSTANCES <= POOL_SIZE:
            raise SystemExit("MIN_INSTANCES must be between 1 and POOL_SIZE")
        self.store = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.store.ping()
        self.api = load_kubernetes()
        self.roles = {
            f"qwen-instance-{index:02d}": ROLE_ACTIVE if index <= MIN_INSTANCES else ROLE_WARM
            for index in range(1, POOL_SIZE + 1)
        }
        self.last_scale_at = 0.0
        self.scale_down_streak = 0
        self.pending_up_streak = 0
        self.tracker = ResourceTracker(time.time())

    def write_roles(self) -> None:
        pipe = self.store.pipeline()
        for name, role in self.roles.items():
            pipe.set(f"pod:{name}:role", role)
        pipe.execute()

    def read_metrics(self, names: list[str], now: float) -> tuple[dict, list[str]]:
        pipe = self.store.pipeline()
        for name in names:
            pipe.get(f"pod:{name}:llumnix_freeness")
            pipe.get(f"pod:{name}:heartbeat")
            pipe.get(f"pod:{name}:running_requests")
            pipe.get(f"pod:{name}:waiting_requests")
            pipe.get(f"pod:{name}:llumnix_scheduler_freeness")
            pipe.get(f"pod:{name}:llumnix_scheduler_heartbeat")
            pipe.get(f"pod:{name}:llumnix_scheduler_running")
            pipe.get(f"pod:{name}:llumnix_scheduler_pending")
        values = pipe.execute()
        metrics = {}
        missing = []
        for index, name in enumerate(names):
            base = index * 8
            side_free, heartbeat, running, waiting = values[base:base + 4]
            sched_free, sched_heartbeat, sched_running, sched_pending = values[base + 4:base + 8]
            try:
                side_fresh = heartbeat is not None and now - float(heartbeat) <= SIDECAR_HEARTBEAT_TTL
            except ValueError:
                side_fresh = False
            try:
                scheduler_fresh = sched_heartbeat is not None and now - float(sched_heartbeat) <= 5.0
            except ValueError:
                scheduler_fresh = False
            if not side_fresh and not scheduler_fresh:
                missing.append(name)
                continue
            sidecar_freeness = float(side_free) if side_fresh and side_free is not None else None
            scheduler_freeness = (
                float(sched_free) if scheduler_fresh and sched_free is not None else None
            )
            available_freeness = [
                value for value in (sidecar_freeness, scheduler_freeness) if value is not None
            ]
            metrics[name] = {
                # Respect both real KV-cache pressure and client-side queued
                # virtual usage. The smaller value is the safer capacity view.
                "freeness": min(available_freeness) if available_freeness else None,
                "sidecar_freeness": sidecar_freeness,
                "running": int(float(sched_running if scheduler_fresh and sched_running is not None else running or 0)),
                "waiting": int(float(sched_pending if scheduler_fresh and sched_pending is not None else waiting or 0)),
                "source": "min(sidecar,scheduler)" if len(available_freeness) == 2
                else "scheduler" if scheduler_freeness is not None else "sidecar",
            }
        return metrics, missing

    def initialize(self) -> dict:
        LOGGER.info("waiting for %d physical KServe instances", POOL_SIZE)
        instances = wait_for_pool(self.api, POOL_SIZE, NAMESPACE)
        keys = list(self.store.scan_iter("llumnix:*"))
        if keys:
            self.store.delete(*keys)
        write_instances(instances)
        self.write_roles()
        LOGGER.info("runtime endpoint file written to %s", RUNTIME_INSTANCES)
        LOGGER.info("initial plan active=%d warm=%d", MIN_INSTANCES, POOL_SIZE - MIN_INSTANCES)

        LOGGER.info(
            "controller startup does not block on sidecar metrics; "
            "load-batch-calary.py will publish virtual freeness"
        )
        return instances

    def apply_plan(self, instances: dict) -> None:
        now = time.time()
        names = sorted(instances)
        metrics, missing = self.read_metrics(names, now)
        action = None

        for name in names:
            if self.roles[name] != ROLE_DRAINING or name not in metrics:
                continue
            if metrics[name]["running"] == 0 and metrics[name]["waiting"] == 0:
                self.roles[name] = ROLE_WARM
                action = f"scale_down_complete:{name}"

        expected_active = [name for name in names if self.roles[name] == ROLE_ACTIVE]
        active = [
            name for name in expected_active
            if name in metrics and metrics[name]["freeness"] is not None
        ]
        freeness = [metrics[name]["freeness"] for name in active]
        pending = sum(metrics[name]["waiting"] for name in active)
        pending_threshold = max(1, len(expected_active) * PENDING_PER_INSTANCE_UP)
        queue_pressure = pending >= pending_threshold
        self.pending_up_streak = self.pending_up_streak + 1 if queue_pressure else 0
        sustained_queue_pressure = self.pending_up_streak >= PENDING_HOLD
        average = sum(freeness) / len(freeness) if freeness else None
        cooling_down = now - self.last_scale_at < SCALE_COOLDOWN
        draining = any(role == ROLE_DRAINING for role in self.roles.values())

        active_metrics_ready = len(active) == len(expected_active)
        if average is not None and active_metrics_ready and not cooling_down and not draining:
            active_or_draining = sum(
                role in {ROLE_ACTIVE, ROLE_DRAINING} for role in self.roles.values()
            )
            if (
                average < SCALE_UP_FREENESS or sustained_queue_pressure
            ) and active_or_draining < POOL_SIZE:
                candidate = next(name for name in names if self.roles[name] == ROLE_WARM)
                self.roles[candidate] = ROLE_ACTIVE
                self.last_scale_at = now
                self.scale_down_streak = 0
                self.pending_up_streak = 0
                action = f"scale_up:{candidate}"
            elif average > SCALE_DOWN_FREENESS and active_or_draining > MIN_INSTANCES:
                self.scale_down_streak += 1
                if self.scale_down_streak >= SCALE_DOWN_HOLD:
                    candidates = [name for name in active if self.roles[name] == ROLE_ACTIVE]
                    victim = min(
                        candidates,
                        key=lambda name: (
                            metrics[name]["running"], metrics[name]["waiting"], -metrics[name]["freeness"], name,
                        ),
                    )
                    self.roles[victim] = ROLE_DRAINING
                    self.last_scale_at = now
                    self.scale_down_streak = 0
                    action = f"scale_down_start:{victim}"
            else:
                self.scale_down_streak = 0

        self.write_roles()
        resource = self.tracker.update(now, self.roles, action)
        plan = {
            "timestamp": now,
            "active": sum(role == ROLE_ACTIVE for role in self.roles.values()),
            "draining": sum(role == ROLE_DRAINING for role in self.roles.values()),
            "warm": sum(role == ROLE_WARM for role in self.roles.values()),
            "freeness_average": average,
            "freeness_min": min(freeness) if freeness else None,
            "freeness_max": max(freeness) if freeness else None,
            "pending_requests": pending,
            "pending_threshold": pending_threshold,
            "pending_streak": self.pending_up_streak,
            "metric_instances": len(metrics),
            "active_metric_instances": len(active),
            "missing_metrics": missing,
            "action": action,
        }
        pipe = self.store.pipeline()
        pipe.set("llumnix:last_plan", json.dumps(plan, ensure_ascii=False), ex=5)
        pipe.set("llumnix:resource", json.dumps(resource, ensure_ascii=False), ex=10)
        pipe.set("llumnix:controller_ready", now, ex=5)
        pipe.set("llumnix:instances_file", str(RUNTIME_INSTANCES), ex=5)
        pipe.execute()

        LOGGER.info(
            "plan A=%d D=%d W=%d active_metrics=%d/%d pending=%d/%d streak=%d freeness(avg/min/max)=%s action=%s",
            plan["active"], plan["draining"], plan["warm"], len(active), len(expected_active),
            pending, pending_threshold, self.pending_up_streak,
            "n/a" if average is None else f"{average:.1f}/{min(freeness):.1f}/{max(freeness):.1f}",
            action or "none",
        )

    def run(self) -> None:
        instances = self.initialize()
        LOGGER.info(
            "controller ready; min=%d max=%d scale_up<%.1f "
            "pending_up>=%d/active for %d intervals scale_down>%.1f",
            MIN_INSTANCES,
            POOL_SIZE,
            SCALE_UP_FREENESS,
            PENDING_PER_INSTANCE_UP,
            PENDING_HOLD,
            SCALE_DOWN_FREENESS,
        )
        while True:
            started = time.time()
            try:
                instances = wait_for_pool(self.api, POOL_SIZE, NAMESPACE, poll_seconds=1.0)
                write_instances(instances)
                self.apply_plan(instances)
            except Exception:
                LOGGER.exception("controller loop failed")
            time.sleep(max(0.0, CONTROL_INTERVAL - (time.time() - started)))


if __name__ == "__main__":
    Controller().run()
