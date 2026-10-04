from __future__ import annotations

import argparse
import asyncio
import json

from aiohttp import web


async def completions(request: web.Request) -> web.StreamResponse:
    _ = await request.json()
    response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    for token in ("hello", " from", " mock"):
        event = {"choices": [{"text": token}]}
        await response.write(f"data: {json.dumps(event)}\n\n".encode())
        await asyncio.sleep(0.003)
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


async def generate(request: web.Request) -> web.StreamResponse:
    payload = await request.json()
    prompt = payload["prompt"]
    response = web.StreamResponse(status=200, headers={"Content-Type": "application/octet-stream"})
    await response.prepare(request)
    cumulative = prompt
    for token in (" hello", " from", " mock"):
        cumulative += token
        await response.write((json.dumps({"text": [cumulative]}) + "\0").encode())
        await asyncio.sleep(0.003)
    await response.write_eof()
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/generate", generate)
    app.router.add_get("/health", lambda _: web.Response(text="ok"))
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()

