"""Llumnix queue-adapted per-instance freeness reporter (llumlet analogue)."""

from __future__ import annotations

import logging
import os
import socket
import sys
import tempfile
import time
from logging.handlers import RotatingFileHandler

import redis
import requests


raw_pod_name = os.getenv("POD_NAME", socket.gethostname())
POD_NAME = raw_pod_name.split("-predictor-")[0]
TARGET_HOST = os.getenv("TARGET_HOST", "127.0.0.1")
TARGET_PORT = int(os.getenv("TARGET_PORT", "8081"))
METRICS_URL = f"http://{TARGET_HOST}:{TARGET_PORT}/metrics"
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CAPACITY_TOKENS = int(os.getenv("INSTANCE_CAPACITY_TOKENS", "8192"))
INTERVAL = float(os.getenv("METRIC_INTERVAL_SECONDS", "1"))


def make_logger() -> logging.Logger:
    log_dir = os.getenv("LOG_DIR", tempfile.gettempdir())
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("LlumnixSidecar")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f"sidecar-{POD_NAME}.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=1,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


LOGGER = make_logger()
REDIS = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def scrape() -> dict:
    response = requests.get(METRICS_URL, timeout=2)
    response.raise_for_status()
    values = {"gpu_cache": 0.0, "running": 0.0, "waiting": 0.0}
    wanted = {
        "vllm:gpu_cache_usage_perc": "gpu_cache",
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
    }
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0].split("{")[0]
        if key in wanted:
            try:
                values[wanted[key]] = float(parts[-1])
            except ValueError:
                pass
    return values


def publish(values: dict) -> None:
    active = max(1.0, values["running"] + values["waiting"])
    physical_tokens = min(CAPACITY_TOKENS, max(0.0, values["gpu_cache"] * CAPACITY_TOKENS))
    freeness = (CAPACITY_TOKENS - physical_tokens) / active
    now = time.time()
    metrics = {
        "gpu_cache_usage": values["gpu_cache"],
        "running_requests": int(values["running"]),
        "waiting_requests": int(values["waiting"]),
        "physical_tokens": physical_tokens,
        "llumnix_freeness": freeness,
        "heartbeat": now,
    }
    pipe = REDIS.pipeline()
    for key, value in metrics.items():
        redis_key = f"pod:{POD_NAME}:{key}"
        pipe.set(redis_key, value)
        pipe.expire(redis_key, 15)
    pipe.execute()
    LOGGER.info(
        "[Llumnix] freeness=%.1f physical=%.1f running=%d waiting=%d",
        freeness, physical_tokens, metrics["running_requests"], metrics["waiting_requests"],
    )


def main() -> None:
    REDIS.ping()
    LOGGER.info("Llumnix queue sidecar ready for %s; scraping %s", POD_NAME, METRICS_URL)
    while True:
        try:
            publish(scrape())
        except Exception as exc:
            LOGGER.warning("metric scrape failed: %s", exc)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
