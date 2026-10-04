import sys
import os
import json
import time
import math
import uuid
import asyncio
import socket
import numpy as np
import requests
import uvicorn
import redis
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
from collections import deque
import tiktoken
import logging
from logging.handlers import RotatingFileHandler

# ==========================================
# 1. 初始化部分 (保持不变)
# ==========================================
_raw_pod_name = os.getenv("POD_NAME", socket.gethostname())
if "-predictor-" in _raw_pod_name:
    POD_NAME = _raw_pod_name.split("-predictor-")[0]
else:
    POD_NAME = _raw_pod_name

print(f"[Init] Logical Instance Name: {POD_NAME}", file=sys.stderr)


def setup_persistent_logger():
    LOG_DIR = os.getenv("LOG_DIR", "/mnt/logs")
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except PermissionError:
        return logging.getLogger("RDP_Metrics")

    filename = f"sidecar-{POD_NAME}.log"
    filepath = os.path.join(LOG_DIR, filename)
    logger = logging.getLogger("RDP_Metrics")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()
    try:
        file_handler = RotatingFileHandler(
            filepath, mode='a', maxBytes=10 * 1024 * 1024, backupCount=1, encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)
    except Exception as e:
        print(f"[Error] Failed to setup logger: {e}", file=sys.stderr)
    return logger


metric_logger = setup_persistent_logger()

try:
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENC = None


def get_token_len(text: str) -> int:
    if ENC:
        try:
            return len(ENC.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 3)


TARGET_HOST = os.getenv("TARGET_HOST", "127.0.0.1")
TARGET_PORT = os.getenv("TARGET_PORT", "80")
SERVICE_URL = f"http://{TARGET_HOST}:{TARGET_PORT}/v1/completions"

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# ==========================================
# 全局配置: 适配 Chiron 本地自动伸缩器参数
# ==========================================
ITL_SLO_MS = float(os.getenv("ITL_SLO_MS", "50"))
# 不再使用 TOKEN_BUDGET，改为 BATCH_SIZE
INIT_BATCH_SIZE = float(os.getenv("INIT_BATCH_SIZE", "20"))
MIN_BATCH_SIZE = float(os.getenv("MIN_BATCH_SIZE", "1.0"))
# Chiron 论文中的 smoothing factor alpha
CHIRON_ALPHA = float(os.getenv("CHIRON_ALPHA", "0.5"))

request_queue = asyncio.Queue()
pending_futures: Dict[str, asyncio.Future] = {}


class GlobalState:
    def __init__(self):
        self.autoscaler_ref = None
        self.current_running_reqs = 0
        self.last_p95_itl = 0.0
        self.last_p95_ttft = 0.0


global_state = GlobalState()


class ThroughputTracker:
    def __init__(self, max_window_seconds=60):
        self.max_window_seconds = max_window_seconds
        self.history = deque()

    def add(self, token_count):
        now = time.time()
        self.history.append((now, token_count))
        while self.history and (now - self.history[0][0] > self.max_window_seconds):
            self.history.popleft()

    def get_tps(self, window_seconds):
        now = time.time()
        limit_time = now - window_seconds
        total_tokens = sum(count for ts, count in self.history if ts > limit_time)
        return total_tokens / float(window_seconds)

    def get_stats(self, window_seconds):
        now = time.time()
        limit_time = now - window_seconds
        valid_records = [(ts, count) for ts, count in self.history if ts > limit_time]
        total_tokens = sum(count for _, count in valid_records)
        if not valid_records:
            actual_duration = 1.0
        else:
            oldest_ts = valid_records[0][0]
            actual_duration = max(1.0, now - oldest_ts)
        divisor = window_seconds if actual_duration >= window_seconds else actual_duration
        return total_tokens, total_tokens / float(divisor)


throughput_tracker = ThroughputTracker(max_window_seconds=60)

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
except Exception as e:
    redis_client = None


# ==========================================
# 2. 上报循环 (适配 Chiron Redis 结构)
# ==========================================
async def metric_reporter_loop():
    print("[Sidecar] Chiron Metric Reporter Started.", file=sys.stderr)
    last_stat_time = time.time()
    STAT_INTERVAL = 10.0

    while True:
        try:
            if redis_client:
                # 1. 获取当前的 Max Batch Size (由 Chiron 算法计算)
                current_max_bs = INIT_BATCH_SIZE
                if global_state.autoscaler_ref:
                    current_max_bs = global_state.autoscaler_ref.current_bs

                # 2. 获取 Running Requests (注意：需确保 GlobalState 已更新为存储 reqs 而非 tokens)
                # 默认值给 0 防止未初始化报错
                running_reqs = getattr(global_state, 'current_running_reqs', 0)

                # 3. 获取 Queue Length
                queue_len = request_queue.qsize()

                # 4.  计算总负载 (正在运行的请求 + 排队的请求)
                total_load_reqs = running_reqs + queue_len

                # 计算 HOL (Head-of-Line) 等待时间
                hol_wait_ms = 0.0
                if queue_len > 0:
                    # 访问队列内部获取第一个请求的到达时间
                    raw_queue = request_queue._queue
                    first_arrival_time = raw_queue[0][3]
                    hol_wait_ms = (time.time() - first_arrival_time) * 1000.0

                # 吞吐量统计
                throughput_short = throughput_tracker.get_tps(10)
                throughput_long = throughput_tracker.get_tps(60)
                tokens_count_60s, _ = throughput_tracker.get_stats(60)

                # 获取 P95 指标 
                time_since_last_batch = time.time() - getattr(global_state, 'last_update_time', 0)
                if time_since_last_batch > 30:
                    # 如果太久没处理请求，指标置 0 避免陈旧数据误导
                    current_itl = 0.0
                    current_ttft = 0.0
                else:
                    current_itl = global_state.last_p95_itl
                    current_ttft = global_state.last_p95_ttft

                # --- Redis Pipeline ---
                pipe = redis_client.pipeline()

                # 写入 Max Batch Size (容量 Capacity)
                pipe.set(f"pod:{POD_NAME}:max_batch_size", current_max_bs)

                # 写入 Total Load Requests (负载 Load = Running + Queue)
                pipe.set(f"pod:{POD_NAME}:total_load_reqs", total_load_reqs)

                # 保留 queue_len 用于独立监控或调试
                pipe.set(f"pod:{POD_NAME}:queue_len", queue_len)

                # 其他常规指标
                pipe.set(f"pod:{POD_NAME}:hol_wait_ms", hol_wait_ms)
                pipe.set(f"pod:{POD_NAME}:p95_itl", float(current_itl))
                pipe.set(f"pod:{POD_NAME}:p95_ttft", float(current_ttft))
                pipe.set(f"pod:{POD_NAME}:throughput_short", throughput_short)
                pipe.set(f"pod:{POD_NAME}:throughput_long", throughput_long)
                pipe.set(f"pod:{POD_NAME}:throughput_count_60s", tokens_count_60s)

                # 设置过期时间 (包括新的 total_load_reqs)
                keys = ["max_batch_size", "total_load_reqs", "queue_len", "hol_wait_ms",
                        "p95_itl", "p95_ttft", "throughput_short", "throughput_long",
                        "throughput_count_60s"]
                for k in keys:
                    pipe.expire(f"pod:{POD_NAME}:{k}", 15)

                pipe.execute()

                # 日志输出：展示 LoadSum (Run + Que)
                metric_logger.info(
                    f"[RDP-Report] BS:{current_max_bs:.1f} | "
                    f"LoadSum:{total_load_reqs} (Run:{running_reqs}+Que:{queue_len}) | "
                    f"ITL:{current_itl:.1f}ms | TTFT:{current_ttft:.1f}ms | "
                    f"HOL:{hol_wait_ms:.0f}ms | TPS:{throughput_short:.1f}"
                )

                now = time.time()
                if now - last_stat_time >= STAT_INTERVAL:
                    metric_logger.info(
                        f"[Throughput-Stat] Window:60s | Count:{tokens_count_60s} toks | "
                        f"Avg TPS:{throughput_long:.1f} toks/s (Output Only)"
                    )
                    last_stat_time = now

        except Exception as e:
            print(f"[Reporter Error] {e}", file=sys.stderr)

        await asyncio.sleep(5)


# ==========================================
# 3. Chiron 本地自动伸缩器
# ==========================================
# 论文引用: Section 4, Algorithm 1
class ChironLocalAutoscaler:
    def __init__(self, itl_slo_ms: float, init_bs: float, min_bs: float, alpha: float):
        self.itl_slo_ms = itl_slo_ms
        self.current_bs = float(init_bs)
        self.min_bs = min_bs
        self.alpha = alpha

        # 记录上一次的吞吐量 (用于计算 TBP)
        self.prev_throughput = 0.0

    def decide_next_batch_size(self, observed_itl: float, observed_throughput: float) -> int:
        """
        根据 Algorithm 1 计算下一个 Batch Size
        """
        # [cite_start]1. 计算 Latency-based Backpressure (LBP) [cite: 236]
        # LBP = ITL / ITL_SLO
        lbp = observed_itl / self.itl_slo_ms

        # [cite_start]2. 计算 Throughput-based Backpressure (TBP) [cite: 241]
        # TBP = Throughput_prev / Throughput_curr
        # 防止除以零
        if observed_throughput > 0 and self.prev_throughput > 0:
            tbp = self.prev_throughput / observed_throughput
        else:
            tbp = 0.0  # 初始阶段或无吞吐时不产生TBP

        # 3. 计算 Local Backpressure
        # Local Backpressure = max(LBP, TBP)
        local_backpressure = max(lbp, tbp)

        # 保存当前吞吐量供下次使用
        self.prev_throughput = observed_throughput

        old_bs = self.current_bs

        # 4. 调整 Batch Size
        if local_backpressure < 1.0:
            # === Scale Up ===
            # 论文 Algorithm 1 Line 9 & 10 逻辑的 Python 实现
            # 逻辑: Batch Size 增加量与 Backpressure 成反比 (BP越小加的越快)
            # 使用 alpha 进行加权增加

            # 防止 backpressure 过小导致除零爆炸 (虽然 <1 但可能接近0)
            safe_bp = max(local_backpressure, 0.01)

            # 增加项: alpha * (1 / BP)
            increment = self.alpha * (1.0 / safe_bp)

            self.current_bs = self.current_bs + increment
        else:
            # === Scale Down ===
            # 论文 Algorithm 1 Line 13: Max Batch Size = Max Batch Size / 2
            self.current_bs = max(self.min_bs, self.current_bs / 2.0)

        return int(self.current_bs)


# ==========================================
# 4. 请求发送代理 (保持不变)
# ==========================================
def send_one_stream(item, headers):
    req_id, prompt_text = item[0], item[1]
    payload = {
        "model": os.getenv("MODEL_NAME", "Qwen-0.5B"),
        "prompt": prompt_text,
        "stream": True,
        "max_tokens": 100,
        "min_tokens": 100,
        "ignore_eos": False,
        "temperature": 0.7
    }
    token_itls = []
    ttft_ms = 0.0
    full_response_text = ""
    error_msg = None
    start_req_time = time.perf_counter()

    try:
        with requests.post(SERVICE_URL, json=payload, headers=headers, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            last_arrival = time.perf_counter()
            is_first_token = True
            for line in resp.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[6:]
                        if json_str.strip() == "[DONE]": break
                        try:
                            chunk = json.loads(json_str)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                text_chunk = chunk['choices'][0]['text']
                                full_response_text += text_chunk
                            now = time.perf_counter()
                            if is_first_token:
                                ttft_ms = (now - start_req_time) * 1000.0
                                is_first_token = False
                            else:
                                itl_ms = (now - last_arrival) * 1000.0
                                token_itls.append(itl_ms)
                            last_arrival = now
                        except Exception:
                            continue
    except Exception as e:
        error_msg = str(e)

    response_data = {"id": req_id, "created": int(time.time()), "choices": [{"text": full_response_text}]}
    if error_msg: response_data["error"] = error_msg
    return req_id, response_data, token_itls, ttft_ms


def send_batch_proxy(batch_items):
    headers = {"Content-Type": "application/json", "X-Accel-Buffering": "no"}
    all_batch_itls = []
    all_batch_ttfts = []
    results_map = {}
    with ThreadPoolExecutor(max_workers=len(batch_items)) as executor:
        future_to_item = {executor.submit(send_one_stream, item, headers): item for item in batch_items}
        for future in as_completed(future_to_item):
            try:
                req_id, resp_data, itls, ttft = future.result()
                results_map[req_id] = resp_data
                all_batch_itls.extend(itls)
                if ttft > 0: all_batch_ttfts.append(ttft)
            except Exception:
                pass
    current_p95_ttft = np.percentile(all_batch_ttfts, 95) if all_batch_ttfts else 0.0
    if all_batch_itls:
        current_p95_itl = np.percentile(all_batch_itls, 95)
    elif all_batch_ttfts:
        current_p95_itl = current_p95_ttft
    else:
        current_p95_itl = float(os.getenv("ITL_SLO_MS", "25.0"))
    return results_map, current_p95_itl, current_p95_ttft


# ==========================================
# 5. 主调度循环
# ==========================================
async def processor_loop():
    print("[Sidecar] Chiron Local Scheduler Started.", file=sys.stderr)

    #  使用新的 Chiron 伸缩器
    autoscaler = ChironLocalAutoscaler(ITL_SLO_MS, INIT_BATCH_SIZE, MIN_BATCH_SIZE, CHIRON_ALPHA)

    # 关联到全局状态
    global_state.autoscaler_ref = autoscaler

    # 初始目标 BS
    target_bs = int(INIT_BATCH_SIZE)
    leftover_item = None

    while True:
        # --- 1. 组装 Batch (基于请求数量限制，而非 Token Budget) ---
        batch_items = []
        current_batch_tokens = 0  # 仅用于统计，不再用于截断
        batch_deadline = None

        while True:
            item = None
            if leftover_item:
                item = leftover_item
                leftover_item = None
                batch_deadline = time.perf_counter() + 0.05
            else:
                if len(batch_items) == 0:
                    item = await request_queue.get()
                    batch_deadline = time.perf_counter() + 0.05
                else:
                    if time.perf_counter() > batch_deadline:
                        break
                    try:
                        item = request_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            # 判断逻辑变更: 检查 Batch Size (Item 个数)
            if len(batch_items) >= target_bs:
                leftover_item = item
                break

            batch_items.append(item)
            # 记录 Token 以供 GlobalState 监控
            current_batch_tokens += (item[2] + 100)

        # 更新全局状态：记录当前正在运行的 Request 个数
        global_state.current_running_reqs = len(batch_items)

        # --- 2. 执行推理 ---
        loop = asyncio.get_running_loop()
        batch_start = time.perf_counter()
        results_map, p95_itl, p95_ttft = await loop.run_in_executor(None, send_batch_proxy, batch_items)
        batch_duration = time.perf_counter() - batch_start

        global_state.last_p95_itl = p95_itl
        global_state.last_p95_ttft = p95_ttft
        #  推理结束，重置为 0
        global_state.current_running_reqs = 0
        global_state.last_update_time = time.time()

        # --- 3. 结果处理 ---
        total_generated_tokens = 0
        error_count = 0
        total_input_tokens = sum(item[2] for item in batch_items)
        current_time = time.time()
        avg_queue_latency = sum((current_time - item[3]) for item in batch_items) / len(
            batch_items) * 1000.0 if batch_items else 0

        for req_id, resp_data in results_map.items():
            if req_id in pending_futures:
                if not pending_futures[req_id].done():
                    pending_futures[req_id].set_result(resp_data)
                del pending_futures[req_id]
            if 'error' in resp_data:
                error_count += 1
                metric_logger.error(f"[Req-Error] ID:{req_id} Msg:{resp_data['error']}")
            if 'choices' in resp_data:
                total_generated_tokens += get_token_len(resp_data['choices'][0]['text'])

        throughput_tracker.add(total_generated_tokens)

        # --- 4. Chiron 决策与日志 ---
        safe_duration = max(batch_duration, 0.001)
        observed_throughput = total_generated_tokens / safe_duration
        if p95_itl <= 0: p95_itl = 0.1

        old_bs = target_bs

        #  调用 Chiron 算法获取新的 Batch Size
        target_bs = autoscaler.decide_next_batch_size(p95_itl, observed_throughput)

        # 获取精确的 float 值用于日志展示
        float_bs = autoscaler.current_bs

        metric_logger.info(
            f"[Batch-Log] "
            f"BS:{old_bs}->{target_bs} (val:{float_bs:.2f}) | "
            f"ITL:{p95_itl:.1f}ms | "
            f"TTFT:{p95_ttft:.1f}ms | "
            f"QTime:{avg_queue_latency:.0f}ms | "
            f"Tput:{observed_throughput:.1f} | "
            f"GenTok:{total_generated_tokens} | "
            f"Size:{len(batch_items)} | "
            f"Dur:{batch_duration * 1000:.0f}ms"
        )


# ==========================================
# 6. Web Server (保持不变)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    t1 = asyncio.create_task(processor_loop())
    t2 = asyncio.create_task(metric_reporter_loop())
    yield
    t1.cancel()
    t2.cancel()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/completions")
@app.post("/openai/v1/completions")
async def handle_request(request: Request):
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        req_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        pending_futures[req_id] = future
        token_count = await loop.run_in_executor(None, get_token_len, prompt)
        arrival_time = time.time()
        await request_queue.put((req_id, prompt, token_count, arrival_time))
        return await future
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, access_log=False)