from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from .metrics import RequestMetric
from .token_utils import count_tokens


def _extract_text(event: Dict[str, Any]) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if isinstance(choice.get("text"), str):
        return choice["text"]
    delta = choice.get("delta") or {}
    return delta.get("content") or ""


async def send_streaming_completion(
    session,
    endpoint: str,
    payload: Dict[str, Any],
    request_id: str,
    service_class: str,
    arrival_monotonic: float,
    scheduler_metadata: Optional[Dict[str, Any]] = None,
) -> RequestMetric:
    """Send an OpenAI-compatible request and measure timestamps from SSE chunks."""
    started = time.perf_counter()
    queue_ms = max(0.0, (started - arrival_monotonic) * 1000.0)
    prompt = payload.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(str(item) for item in prompt)
    input_tokens = count_tokens(str(prompt))
    token_times = []
    pieces = []
    status = None
    error = None

    try:
        async with session.post(endpoint, json=payload) as response:
            status = response.status
            if response.status >= 400:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:500]}")

            buffer = b""
            async for chunk in response.content.iter_any():
                buffer += chunk
                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    line = raw_line.strip()
                    if not line or not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    try:
                        event = json.loads(data.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    text = _extract_text(event)
                    if text:
                        pieces.append(text)
                        token_times.append(time.perf_counter())

            # Some compatible servers return one non-streaming JSON response.
            if not pieces and buffer.strip():
                try:
                    event = json.loads(buffer.decode("utf-8"))
                    text = _extract_text(event)
                    if text:
                        pieces.append(text)
                        token_times.append(time.perf_counter())
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
    except Exception as exc:  # caller needs a record even for failures
        error = str(exc)

    ended = time.perf_counter()
    output_text = "".join(pieces)
    ttft_ms = (token_times[0] - started) * 1000.0 if token_times else None
    itl_ms = [(right - left) * 1000.0 for left, right in zip(token_times, token_times[1:])]
    success = error is None and status is not None and 200 <= status < 300
    return RequestMetric(
        request_id=request_id,
        service_class=service_class,
        endpoint=endpoint,
        input_tokens=input_tokens,
        output_tokens=count_tokens(output_text),
        arrival_s=arrival_monotonic,
        queue_ms=queue_ms,
        ttft_ms=ttft_ms,
        itl_ms=itl_ms,
        e2e_ms=(ended - started) * 1000.0,
        status=status,
        success=success,
        error=error,
        scheduler=scheduler_metadata or {},
    )

