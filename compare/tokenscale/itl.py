"""TokenScale per-instance capacity reporter used by the KServe warm pool."""

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
INTERVAL = float(os.getenv("METRIC_INTERVAL_SECONDS", "1"))


def make_logger() -> logging.Logger:
    log_dir = os.getenv("LOG_DIR", tempfile.gettempdir())
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("TokenScaleSidecar")
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
LAST = {"time": time.time(), "prompt_tokens": None, "generation_tokens": None,
        "ttft_sum": None, "ttft_count": None, "itl_sum": None, "itl_count": None}


def scrape() -> dict:
    response = requests.get(METRICS_URL, timeout=2)
    response.raise_for_status()
    values = {}
    wanted = {
        "vllm:gpu_cache_usage_perc": "gpu_cache",
        "vllm:num_requests_running": "running_requests",
        "vllm:num_requests_waiting": "waiting_requests",
        "vllm:prompt_tokens_total": "prompt_tokens",
        "vllm:generation_tokens_total": "generation_tokens",
        "vllm:time_to_first_token_seconds_sum": "ttft_sum",
        "vllm:time_to_first_token_seconds_count": "ttft_count",
        "vllm:time_per_output_token_seconds_sum": "itl_sum",
        "vllm:time_per_output_token_seconds_count": "itl_count",
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


def delta_rate(values: dict, key: str, now: float) -> float:
    current = values.get(key)
    previous = LAST.get(key)
    elapsed = max(now - float(LAST["time"]), 1e-9)
    if current is None or previous is None or current < previous:
        return 0.0
    return (current - previous) / elapsed


def delta_average_ms(values: dict, sum_key: str, count_key: str) -> float:
    current_sum, current_count = values.get(sum_key), values.get(count_key)
    previous_sum, previous_count = LAST.get(sum_key), LAST.get(count_key)
    if None in (current_sum, current_count, previous_sum, previous_count):
        return 0.0
    count_delta = current_count - previous_count
    return max(0.0, (current_sum - previous_sum) / count_delta * 1000.0) if count_delta > 0 else 0.0


def publish(values: dict) -> None:
    now = time.time()
    metrics = {
        "gpu_cache_usage": values.get("gpu_cache", 0.0),
        "running_requests": int(values.get("running_requests", 0)),
        "waiting_requests": int(values.get("waiting_requests", 0)),
        "prompt_tps": delta_rate(values, "prompt_tokens", now),
        "generation_tps": delta_rate(values, "generation_tokens", now),
        "avg_ttft_ms": delta_average_ms(values, "ttft_sum", "ttft_count"),
        "avg_itl_ms": delta_average_ms(values, "itl_sum", "itl_count"),
        "heartbeat": now,
    }
    pipe = REDIS.pipeline()
    for key, value in metrics.items():
        redis_key = f"pod:{POD_NAME}:{key}"
        pipe.set(redis_key, value)
        pipe.expire(redis_key, 15)
    pipe.execute()
    for key in ("prompt_tokens", "generation_tokens", "ttft_sum", "ttft_count", "itl_sum", "itl_count"):
        if key in values:
            LAST[key] = values[key]
    LAST["time"] = now
    LOGGER.info(
        "[TokenScale] role=%s input_tps=%.1f output_tps=%.1f running=%d waiting=%d TTFT=%.1fms ITL=%.1fms",
        REDIS.get(f"pod:{POD_NAME}:role") or "WARM",
        metrics["prompt_tps"], metrics["generation_tps"], metrics["running_requests"],
        metrics["waiting_requests"], metrics["avg_ttft_ms"], metrics["avg_itl_ms"],
    )


def main() -> None:
    REDIS.ping()
    LOGGER.info("TokenScale sidecar ready for %s; scraping %s", POD_NAME, METRICS_URL)
    while True:
        try:
            publish(scrape())
        except Exception as exc:
            LOGGER.warning("metric scrape failed: %s", exc)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
