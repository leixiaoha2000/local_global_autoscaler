from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from .token_utils import count_tokens


@dataclass(frozen=True)
class WorkItem:
    offset_s: float
    prompt: str
    service_class: str
    max_tokens: int

    @property
    def input_tokens(self) -> int:
        return count_tokens(self.prompt)


def load_prompts(path: Union[str, Path]) -> List[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    prompts = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompts.append(item["prompt"])
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def load_window_counts(path: Union[str, Path]) -> List[int]:
    counts = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip():
                continue
            try:
                counts.append(max(0, int(float(row[0].strip()))))
            except ValueError:
                continue
    if not counts:
        raise ValueError(f"no numeric workload windows found in {path}")
    return counts


def build_uniform_workload(
    prompts: List[str],
    window_counts: List[int],
    window_seconds: float,
    interactive_ratio: float,
    max_tokens: int,
    seed: int = 2026,
) -> List[WorkItem]:
    if not 0.0 <= interactive_ratio <= 1.0:
        raise ValueError("interactive_ratio must be in [0, 1]")
    rng = random.Random(seed)
    work = []
    sequence = 0
    for window_idx, count in enumerate(window_counts):
        interval = window_seconds / count if count else window_seconds
        for idx in range(count):
            # A deterministic stratified split is less noisy than independent Bernoulli draws.
            fractional = ((sequence * 0.6180339887498949) % 1.0)
            service_class = "interactive" if fractional < interactive_ratio else "batch"
            work.append(WorkItem(
                offset_s=window_idx * window_seconds + idx * interval,
                prompt=rng.choice(prompts),
                service_class=service_class,
                max_tokens=max_tokens,
            ))
            sequence += 1
    return work


def build_per_class_workload(
    prompts: List[str],
    window_counts: List[int],
    window_seconds: float,
    max_tokens: int,
    seed: int = 2026,
) -> List[WorkItem]:
    """Replay every CSV count once for interactive and once for batch.

    This matches the repository's historical setup where the two load-generator
    scripts run concurrently against the same window-count trace.
    """
    rng = random.Random(seed)
    work = []
    for window_idx, count in enumerate(window_counts):
        interval = window_seconds / count if count else window_seconds
        for service_class in ("interactive", "batch"):
            for idx in range(count):
                work.append(WorkItem(
                    offset_s=window_idx * window_seconds + idx * interval,
                    prompt=rng.choice(prompts),
                    service_class=service_class,
                    max_tokens=max_tokens,
                ))
    return sorted(work, key=lambda item: item.offset_s)
