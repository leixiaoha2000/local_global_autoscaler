from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


@dataclass
class RequestMetric:
    request_id: str
    service_class: str
    endpoint: str
    input_tokens: int
    output_tokens: int
    arrival_s: float
    queue_ms: float
    ttft_ms: Optional[float]
    itl_ms: List[float]
    e2e_ms: float
    status: Optional[int]
    success: bool
    error: Optional[str] = None
    scheduler: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_itl_ms(self) -> Optional[float]:
        return statistics.fmean(self.itl_ms) if self.itl_ms else None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["mean_itl_ms"] = self.mean_itl_ms
        return result


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    ordered = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: Iterable[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "mean": statistics.fmean(vals),
        "p50": percentile(vals, 50),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
    }


def _request_meets_slo(record: RequestMetric, slo: Dict[str, Dict[str, float]]) -> bool:
    limits = slo.get(record.service_class, slo.get("default", {}))
    if not record.success or record.ttft_ms is None:
        return False
    if record.ttft_ms > float(limits.get("ttft_ms", math.inf)):
        return False
    mean_itl = record.mean_itl_ms
    return mean_itl is None or mean_itl <= float(limits.get("itl_ms", math.inf))


def summarize(
    records: List[RequestMetric],
    duration_s: float,
    slo: Dict[str, Dict[str, float]],
    resource: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    successful = [r for r in records if r.success]
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    request_itls = [r.mean_itl_ms for r in successful if r.mean_itl_ms is not None]
    all_itls = [gap for r in successful for gap in r.itl_ms]
    attainment = [_request_meets_slo(r, slo) for r in records]
    per_class: Dict[str, Dict[str, Any]] = {}
    for service_class in sorted({r.service_class for r in records}):
        subset = [r for r in records if r.service_class == service_class]
        met = sum(_request_meets_slo(r, slo) for r in subset)
        per_class[service_class] = {
            "requests": len(subset),
            "success": sum(r.success for r in subset),
            "slo_attainment": met / len(subset) if subset else None,
        }

    safe_duration = max(duration_s, 1e-9)
    result = {
        "duration_s": duration_s,
        "requests": len(records),
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "success_rate": len(successful) / len(records) if records else None,
        "input_tokens": sum(r.input_tokens for r in successful),
        "output_tokens": sum(r.output_tokens for r in successful),
        "input_tps": sum(r.input_tokens for r in successful) / safe_duration,
        "output_tps": sum(r.output_tokens for r in successful) / safe_duration,
        "total_tps": sum(r.input_tokens + r.output_tokens for r in successful) / safe_duration,
        "request_qps": len(successful) / safe_duration,
        "ttft_ms": _distribution(ttfts),
        "request_mean_itl_ms": _distribution(request_itls),
        "token_gap_itl_ms": _distribution(all_itls),
        "e2e_ms": _distribution(r.e2e_ms for r in successful),
        "queue_ms": _distribution(r.queue_ms for r in records),
        "slo_attainment": sum(attainment) / len(attainment) if attainment else None,
        "goodput_rps": sum(attainment) / safe_duration,
        "by_service_class": per_class,
        "slo": slo,
    }
    if resource:
        result["resource"] = resource
    return result


def write_jsonl(path: Union[str, Path], records: Iterable[RequestMetric]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
