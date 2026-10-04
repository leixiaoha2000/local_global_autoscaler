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
# 1. 先计算 POD_NAME (必须放在最前面！)
# ==========================================
# 逻辑 Pod 名提取逻辑：
# KServe 部署的 Pod 名字通常是 "qwen-instance-01-predictor-default-xxxxx"
# 我们只需要 "qwen-instance-01" 这一部分作为 Redis Key
_raw_pod_name = os.getenv("POD_NAME", socket.gethostname())
if "-predictor-" in _raw_pod_name:
    POD_NAME = _raw_pod_name.split("-predictor-")[0]
else:
    POD_NAME = _raw_pod_name

print(f"[Init] Logical Instance Name: {POD_NAME}", file=sys.stderr)

# ==========================================
# 2. 再配置持久化日志 (现在可以用 POD_NAME 了)
# ==========================================
def setup_persistent_logger():
    # 1. 确定日志挂载
    LOG_DIR = os.getenv("LOG_DIR", "/mnt/logs")
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except PermissionError:
        #防止因为权限问题直接崩溃，回退到 stderr
        print(f"[Error] No permission to create {LOG_DIR}, fallback to stderr", file=sys.stderr)
        print(f"[Error] No permission to create {LOG_DIR}, fallback to stderr", file=sys.stderr)
        return logging.getLogger("RDP_Metrics") # 返回默认logger

    # 2. 构造唯一文件名字如sidecar-qwen-instance-01.log
    filename = f"sidecar-{POD_NAME}.log"  # <--- 现在这里安全了
    filepath = os.path.join(LOG_DIR, filename)

    logger = logging.getLogger("RDP_Metrics")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 1. 如果已有 handler，先清理（防止重复）
    if logger.handlers:
        logger.handlers.clear()

    # 2. 无论是否清理，都要重新创建 Handler (注意这里的缩进！)
    try:
        # backupCount=1 是实现“容量限制”的最小值
        file_handler = RotatingFileHandler(
            filepath,
            mode='a',
            maxBytes=10 * 1024 * 1024,  # 10MB 触发翻转
            backupCount=1,  # 只保留 1 个旧备份，总占用 20MB
            encoding='utf-8'
        )

        file_fmt = logging.Formatter('%(asctime)s | %(message)s')
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        # 同时也建议保留控制台输出
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)

        print(f"[Init] Logging to file: {filepath}", file=sys.stderr)

    except Exception as e:
        print(f"[Error] Failed to setup logger: {e}", file=sys.stderr)

    return logger

# 初始化全局 Logger
metric_logger = setup_persistent_logger()

# ==========================================
# 1. 全局配置与状态管理
# ==========================================
try:
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    print("[Warning] tiktoken encoding load failed, using fallback.", file=sys.stderr)
    ENC = None


def get_token_len(text: str) -> int:
    if ENC:
        try:
            return len(ENC.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 3)


# 基础配置: 后端 vLLM 的地址
TARGET_HOST = os.getenv("TARGET_HOST", "127.0.0.1")
TARGET_PORT = os.getenv("TARGET_PORT", "80")
SERVICE_URL = f"http://{TARGET_HOST}:{TARGET_PORT}/v1/completions"

# Redis 配置
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))



# GBS 算法参数: 定义了延迟目标 (SLO) 和初始批处理大小 (Budget)
ITL_SLO_MS = float(os.getenv("ITL_SLO_MS", "15"))
INIT_TOKEN_BUDGET = int(os.getenv("INIT_TOKEN_BUDGET", "2048"))
MIN_TOKEN_BUDGET = int(os.getenv("MIN_TOKEN_BUDGET", "128"))
GBS_STEP_RATIO = float(os.getenv("GBS_STEP_RATIO", "0.05"))

# 全局请求队列: Web Server 收到请求放这里，Processor Loop 从这里取 (Item: req_id, prompt, token_count, arrival_time)
request_queue = asyncio.Queue()
# 挂起的 Future: 用于将异步处理结果返回给等待的 Web 请求
pending_futures: Dict[str, asyncio.Future] = {}


# --- [新增] 全局状态共享 ---
# 用于在 Processor Loop (处理线程) 和 Metric Reporter (上报线程) 之间同步数据
class GlobalState:
    def __init__(self):
        self.autoscaler_ref = None  # 指向 GBS 算法实例，用于获取当前 Budget
        self.current_running_tokens = 0  # 当前 Batch 中正在运行的 Token 总数
        self.last_p95_itl = 0.0  # [新增] 最近一次观测到的 P95 ITL
        self.last_p95_ttft = 0.0


global_state = GlobalState()


# 吞吐量追踪器:滑动窗口计算 TPS (Tokens Per Second)
class ThroughputTracker:
    def __init__(self, max_window_seconds=60):
        self.max_window_seconds = max_window_seconds
        self.history = deque()

    def add(self, token_count):
        """记录一次生成的 Token 数量"""
        now = time.time()
        self.history.append((now, token_count))
        # 移除过期数据
        while self.history and (now - self.history[0][0] > self.max_window_seconds):
            self.history.popleft()

    def get_tps(self, window_seconds):
        now = time.time()
        limit_time = now - window_seconds
        total_tokens = sum(count for ts, count in self.history if ts > limit_time)
        return total_tokens / float(window_seconds)

    def get_stats(self, window_seconds):
        """
        [新增] 获取指定窗口内的统计详情
        返回: (total_tokens, tps)
        """
        now = time.time()
        limit_time = now - window_seconds

        # 筛选出窗口内的记录
        valid_records = [(ts, count) for ts, count in self.history if ts > limit_time]
        total_tokens = sum(count for _, count in valid_records)

        # 计算时间跨度：如果历史记录不足 window_seconds (刚启动)，则用实际跨度，避免 TPS 虚低
        if not valid_records:
            actual_duration = 1.0
        else:
            # 这种计算方式在刚启动时更准确：max(1.0, now - 最早记录时间)
            # 但为了指标稳定性，通常还是除以固定窗口 window_seconds，除非刚启动
            # 这里采用稳健策略：刚启动时按实际时间算，稳定后按窗口算
            oldest_ts = valid_records[0][0]
            actual_duration = max(1.0, now - oldest_ts)

        # 混合策略：时间不够窗口长度时用 actual_duration，否则用 window_seconds
        divisor = window_seconds if actual_duration >= window_seconds else actual_duration

        return total_tokens, total_tokens / float(divisor)


throughput_tracker = ThroughputTracker(max_window_seconds=60)

# Redis 连接
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    print(f"[Init] Redis connected at {REDIS_HOST}:{REDIS_PORT}", file=sys.stderr)
except Exception as e:
    print(f"[Warning] Redis connection failed: {e}", file=sys.stderr)
    redis_client = None


# ==========================================
# 2. 上报循环 (适配 RDP 需求)
# ==========================================
# 后台任务，每 5 秒向 Redis 写入一次当前 Pod 的状态。这些数据是 Global Scaler (全局伸缩器) 进行扩缩容决策的依据
async def metric_reporter_loop():
    """
    向 Redis 上报 RDP 全局伸缩器所需的指标，并打印双流日志（控制流 + 统计流）
    """
    print("[Sidecar] RDP Metric Reporter Started.", file=sys.stderr)

    # [新增] 统计日志的时间戳记录
    last_stat_time = time.time()
    STAT_INTERVAL = 10.0  # 每 10 秒打印一次统计日志

    while True:
        try:
            if redis_client:
                # ===========================
                # 1. 获取各项指标
                # ===========================

                # A. 容量: 当前 GBS 算法允许的最大并发 Token 数 (Budget)
                current_budget = INIT_TOKEN_BUDGET
                if global_state.autoscaler_ref:
                    current_budget = global_state.autoscaler_ref.current_budget

                # B. 负载: 正在跑的 + 在排队的
                running_tokens = global_state.current_running_tokens

                queued_tokens = 0
                hol_wait_ms = 0.0 # 队头等待时间 (Head-Of-Line Wait)
                queue_len = request_queue.qsize()

                if queue_len > 0:
                    raw_queue = request_queue._queue
                    # 计算队列中所有请求的 Token 总和
                    # [修改后] 保持一致，排队时也预留 100 的量
                    queued_tokens = sum((item[2] + 100) for item in raw_queue)
                    # 计算最老请求等待了多久 (用于批处理平面扩容)
                    first_arrival_time = raw_queue[0][3]
                    hol_wait_ms = (time.time() - first_arrival_time) * 1000.0

                # C. 性能: 吞吐量和延迟
                throughput_short = throughput_tracker.get_tps(10)
                throughput_long = throughput_tracker.get_tps(60)

                # [新增] 获取 60s 内的总生成量 (用于日志和 Redis)
                # 注意：需确保 ThroughputTracker 类已实现 get_stats 方法
                tokens_count_60s, _ = throughput_tracker.get_stats(60)

                # D. 延迟 (Latency)
                # [优化] 如果超过 30秒 没有新的 Batch 更新 ITL，说明系统空闲，重置为 0
                time_since_last_batch = time.time() - getattr(global_state, 'last_update_time', 0)
                if time_since_last_batch > 30:
                    current_itl = 0.0
                    current_ttft = 0.0
                else:
                    current_itl = global_state.last_p95_itl
                    current_ttft = global_state.last_p95_ttft

                # ===========================
                # 2. 上报 Redis
                # ===========================
                pipe = redis_client.pipeline()

                # [RDP 核心指标]
                pipe.set(f"pod:{POD_NAME}:budget", current_budget)
                pipe.set(f"pod:{POD_NAME}:running_tokens", running_tokens)
                pipe.set(f"pod:{POD_NAME}:queued_tokens", queued_tokens)
                pipe.set(f"pod:{POD_NAME}:hol_wait_ms", hol_wait_ms)

                # [新增] 上报 P95 ITL (供监控)
                pipe.set(f"pod:{POD_NAME}:p95_itl", float(current_itl))
                pipe.set(f"pod:{POD_NAME}:p95_ttft", float(current_ttft))

                # [辅助指标]
                pipe.set(f"pod:{POD_NAME}:queue_len", queue_len)
                pipe.set(f"pod:{POD_NAME}:throughput_short", throughput_short)
                pipe.set(f"pod:{POD_NAME}:throughput_long", throughput_long)

                # [新增] 上报 60s 总产量 (供大盘展示)
                pipe.set(f"pod:{POD_NAME}:throughput_count_60s", tokens_count_60s)

                # 设置过期时间
                keys = ["budget", "running_tokens", "queued_tokens", "hol_wait_ms",
                        "p95_itl", "p95_ttft", "queue_len", "throughput_short", "throughput_long",
                        "throughput_count_60s"]
                for k in keys:
                    pipe.expire(f"pod:{POD_NAME}:{k}", 15)

                pipe.execute()

                # ===========================
                # 3. 日志打印
                # ===========================

                # 日志流 A: RDP 高频控制日志 (每 5 秒)
                # [修改] 增加了 ITL 字段
                metric_logger.info(
                    f"[RDP-Report] Bud:{current_budget:.0f} | "
                    f"ITL:{current_itl:.1f}ms | TTFT:{current_ttft:.1f}ms | "  # <--- [修改] 日志增加显示
                    f"RunTok:{running_tokens} | QueTok:{queued_tokens} | "
                    f"HOL:{hol_wait_ms:.0f}ms | TPS:{throughput_short:.1f}"
                )

                # 日志流 B: 定期吞吐量统计日志 (每 10 秒)
                now = time.time()
                if now - last_stat_time >= STAT_INTERVAL:
                    # 复用前面获取的 tokens_count_60s 和 throughput_long (即 60s TPS)
                    metric_logger.info(
                        f"[Throughput-Stat] Window:60s | Count:{tokens_count_60s} toks | "
                        f"Avg TPS:{throughput_long:.1f} toks/s (Output Only)"
                    )

                    last_stat_time = now

        except Exception as e:
            print(f"[Reporter Error] {e}", file=sys.stderr)

        await asyncio.sleep(5)


# ==========================================
# 3. GBS 自动伸缩器 (逻辑不变)
# ==========================================
class GBSAutoscaler:
    def __init__(self, itl_slo_ms: float, init_budget: int, min_budget: int, step_ratio: float = 0.05):
        self.itl_slo_ms = itl_slo_ms
        self.min_budget = min_budget
        self.state = "SEARCH" # 状态机: SEARCH (爬山搜索) / RECOVERY (快速恢复)
        self.current_budget = float(init_budget)
        self.last_score = -1.0
        self.direction = 1
        self.eta = step_ratio
        # 定义自适应步长的边界
        self.eta_min = 0.01
        self.eta_max = 0.15
        self.gamma_acc = 1.2
        self.gamma_dec = 0.5
        self.recovery_counter = 0
        # [新增] 用于进入恢复态的计数器
        self.violation_counter = 0
        self.violation_threshold = 3  # 必须连续 3 次违约才熔断
        self.recovery_threshold = 3

    def _calculate_score(self, throughput: float, itl: float) -> float:
        # 奖励函数: 吞吐量越高越好，但如果 ITL 超标，施加平方惩罚
        if itl <= self.itl_slo_ms:
            penalty = 1.0
        else:
            penalty = (self.itl_slo_ms / itl) ** 2
        return throughput * penalty

    def decide_next_budget(self, observed_itl: float, observed_throughput: float) -> int:
        # 1. 紧急恢复模式: 延迟严重超标 (>110% SLO)
        # [修改] 增加去抖动逻辑 (Debouncing)
        if observed_itl > 1.1 * self.itl_slo_ms:
            self.violation_counter += 1

            # 只有连续 N 次超标，才真正触发熔断
            if self.violation_counter >= self.violation_threshold:
                self.current_budget = max(self.min_budget, self.current_budget * 0.8)
                self.state = "RECOVERY"
                self.eta = self.eta_min
                self.direction = 1

                # 重置计数器
                self.recovery_counter = 0
                self.violation_counter = 0
                self.last_score = -1.0

                return int(self.current_budget)
            else:
                # 虽然超标但还没达到阈值，暂时保持现状或轻微抑制
                # 可以选择不更新 Budget，或者仅轻微减小，或者直接返回旧 Budget
                # 这里建议：暂时按兵不动，给它机会自我恢复
                return int(self.current_budget)
        else:
            # [关键] 如果中间有一次正常了，违约计数器清零！
            # 这保证了必须是"连续"的坏蛋
            self.violation_counter = 0
        # 2. 恢复期检测: 如果连续几次都正常了，回到 SEARCH 模式
        if self.state == "RECOVERY":
            if observed_itl <= self.itl_slo_ms:
                self.recovery_counter += 1
                if self.recovery_counter >= self.recovery_threshold:
                    self.state = "SEARCH"
                    self.direction = 1
            else:
                self.recovery_counter = 0
            return int(self.current_budget)
        # 3. 搜索模式 (Hill Climbing / 梯度上升类似逻辑)
        current_score = self._calculate_score(observed_throughput, observed_itl)
        new_direction = self.direction
        # 如果得分比上次低，说明走错方向了，掉头
        if self.last_score > 0:
            if current_score < self.last_score:
                new_direction = -self.direction
            else:
                new_direction = self.direction
        # 动态步长调整
        if self.last_score > 0:
            if new_direction == self.direction:
                self.eta = min(self.eta * self.gamma_acc, self.eta_max)
            else:
                self.eta = max(self.eta * self.gamma_dec, self.eta_min)

        self.direction = new_direction
        # [逻辑修正] 确保最小步长至少为 1，防止 float 运算导致原地踏步
        delta = max(1.0, self.current_budget * self.eta)
        new_budget_val = self.current_budget + (self.direction * delta)
        self.current_budget = max(self.min_budget, new_budget_val)
        self.last_score = current_score
        return int(self.current_budget)


# ==========================================
# 4. 请求发送代理
# ==========================================
def send_one_stream(item, headers):
    # 发送单个请求到 vLLM
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
    ttft_ms = 0.0  # [新增] 初始化 TTFT
    full_response_text = ""
    error_msg = None

    # [新增] 记录请求发出的起始时间
    start_req_time = time.perf_counter()

    try:
        # 建立流式连接
        with requests.post(SERVICE_URL, json=payload, headers=headers, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            last_arrival = time.perf_counter()
            is_first_token = True
            # 逐行解析 SSE
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
                                # 计算 TTFT: 从发出请求到收到第一个 Token
                                ttft_ms = (now - start_req_time) * 1000.0
                                is_first_token = False
                            else:
                                # 计算 ITL: 相邻 Token 之间的时间差
                                itl_ms = (now - last_arrival) * 1000.0
                                token_itls.append(itl_ms)
                            last_arrival = now
                        except Exception:
                            continue
    except Exception as e:
        error_msg = str(e)

    response_data = {
        "id": req_id,
        "created": int(time.time()),
        "choices": [{"text": full_response_text}]  # 始终保留已生成的文本
    }
    if error_msg:
        # 将错误信息注入，而不是覆盖整个对象
        response_data["error"] = error_msg

    return req_id, response_data, token_itls, ttft_ms


def send_batch_proxy(batch_items):
    # 并发执行 Batch 中的所有请求
    headers = {"Content-Type": "application/json", "X-Accel-Buffering": "no"}
    all_batch_itls = []
    all_batch_ttfts = []  # [新增] 收集本批次所有请求的 TTFT
    results_map = {}
    # 使用线程池并发 IO
    with ThreadPoolExecutor(max_workers=len(batch_items)) as executor:
        future_to_item = {executor.submit(send_one_stream, item, headers): item for item in batch_items}
        for future in as_completed(future_to_item):
            try:
                # [修改] 接收 4 个返回值
                req_id, resp_data, itls, ttft = future.result()
                results_map[req_id] = resp_data
                all_batch_itls.extend(itls)

                # [新增] 收集有效 TTFT
                if ttft > 0:
                    all_batch_ttfts.append(ttft)
            except Exception:
                pass

    # 计算本批次的 P95 ITL和TTFT
    # [修正后] 智能回退逻辑
    # 1. 计算 TTFT (TTFT 总是存在的，除非全失败)
    current_p95_ttft = np.percentile(all_batch_ttfts, 95) if all_batch_ttfts else 0.0

    # 2. 计算 ITL
    if all_batch_itls:
        # 正常情况：有生成 Token，计算真实的 ITL
        current_p95_itl = np.percentile(all_batch_itls, 95)
    elif all_batch_ttfts:
        # 特殊情况：全是短请求(无 ITL)，用 TTFT 代替 ITL 作为拥塞信号
        # 注意：这里假设如果 Prefill 慢，说明负载高，算法应该感知到
        current_p95_itl = current_p95_ttft
    else:
        # 极端情况：啥数据都没有 (全失败?)
        # 此时给一个“中性值”或者 SLO 上限，防止算法乱动，但绝不能给 0.1
        current_p95_itl = float(os.getenv("ITL_SLO_MS", "25.0"))

    return results_map, current_p95_itl, current_p95_ttft


# ==========================================
# 5. 主调度循环 (GBS Kernel)
# ==========================================
async def processor_loop():
    print("[Sidecar] GBS Scheduler Started.", file=sys.stderr)
    autoscaler = GBSAutoscaler(ITL_SLO_MS, INIT_TOKEN_BUDGET, MIN_TOKEN_BUDGET, GBS_STEP_RATIO)

    # [关联] 将 autoscaler 实例暴露给 GlobalState 供上报使用 (Redis 用)
    global_state.autoscaler_ref = autoscaler

    target_budget = INIT_TOKEN_BUDGET
    leftover_item = None
    # [新增] 记录上次衰减的时间
    last_decay_time = time.time()
    while True:
        # --- 1. 组装 Batch (Batching) ---
        batch_items = []
        current_batch_tokens = 0
        batch_deadline = None
        #从队列中取请求，直到 Token 总数达到 target_budget
        while True:
            item = None
            if leftover_item:
                item = leftover_item
                leftover_item = None

                # [关键修复] 遗留项也相当于本轮的"第一个请求"，必须启动倒计时！
                # 否则下一轮循环检查超时时会因 NoneType 报错
                batch_deadline = time.perf_counter() + 0.05

            else:
                if len(batch_items) == 0:
                    item = await request_queue.get()
                    # 正常流程的倒计时
                    batch_deadline = time.perf_counter() + 0.05
                else:
                    # 检查超时
                    # 修复后这里 batch_deadline 必定是 float，不会报错
                    if time.perf_counter() > batch_deadline:
                        break
                    try:
                        item = request_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            # 必须定义在循环内部或者循环之前
            FIXED_OUTPUT_LEN = 100
            item_len = item[2]
            if len(batch_items) > 0 and (current_batch_tokens + item_len + FIXED_OUTPUT_LEN > target_budget):
                leftover_item = item
                break

            batch_items.append(item)
            current_batch_tokens += (item_len + FIXED_OUTPUT_LEN)
            if current_batch_tokens >= target_budget: break

        # 更新全局状态: 当前这批次正在跑多少 Token
        global_state.current_running_tokens = current_batch_tokens

        # --- 2. 执行推理 (Batch 维度) ---
        loop = asyncio.get_running_loop()
        batch_start = time.perf_counter()

        # [核心计算] 这里的 p95_itl 是基于当前这一个 Batch 内所有 Token 算出来的精准值
        results_map, p95_itl, p95_ttft = await loop.run_in_executor(None, send_batch_proxy, batch_items)

        batch_duration = time.perf_counter() - batch_start

        # [状态同步] 更新给 GlobalState，供 Redis 定时上报使用 (保持心跳)
        global_state.last_p95_itl = p95_itl
        global_state.last_p95_ttft = p95_ttft
        global_state.current_running_tokens = 0
        # [关键修复] 必须更新这个时间戳，否则 Reporter 会认为数据过期而归零
        global_state.last_update_time = time.time()

        # --- 3. 结果处理与吞吐统计 ---
        total_generated_tokens = 0
        error_count = 0  # [新增] 统计错误数
        # [新增] 计算本批次的输入 Token 总量
        total_input_tokens = sum(item[2] for item in batch_items)

        # [新增] 计算本批次的平均排队时间
        current_time = time.time()
        # item[3] 是我们在 handle_request 里放入的 arrival_time
        avg_queue_latency = sum((current_time - item[3]) for item in batch_items) / len(batch_items) * 1000.0
        for req_id, resp_data in results_map.items():
            if req_id in pending_futures:
                if not pending_futures[req_id].done():
                    pending_futures[req_id].set_result(resp_data)
                del pending_futures[req_id]
            if 'error' in resp_data:  # [新增] 检查错误
                error_count += 1
                # 可选：如果报错，打印具体 Error Log
                metric_logger.error(f"[Req-Error] ID:{req_id} Msg:{resp_data['error']}")
            if 'choices' in resp_data:
                total_generated_tokens += get_token_len(resp_data['choices'][0]['text'])

        throughput_tracker.add(total_generated_tokens)

        # --- 4. GBS 决策与日志输出 ---
        safe_duration = max(batch_duration, 0.001)
        observed_throughput = total_generated_tokens / safe_duration

        # 修正无效值
        if p95_itl <= 0: p95_itl = 0.1

        # ================= [修改版：移除保护机制] =================
        # 1. 记录旧 Budget（用于日志对比）
        old_budget = target_budget

        # 2. 计算饱和度 (仅用于日志记录，不再用于控制)
        current_total_load = sum(item[2] + FIXED_OUTPUT_LEN for item in batch_items)
        saturation_rate = current_total_load / target_budget if target_budget > 0 else 0


        # 直接调用算法更新 Budget
        target_budget = autoscaler.decide_next_budget(p95_itl, observed_throughput)
        decision_msg = "Update"

        # 后果警告 2：此处没有 Decay 逻辑。
        # 当系统空闲时，Budget 将保持在最后一次 Update 的值（可能是高位）。
        # 下次突发流量到来时可能引发 OOM。

        # 4. 统一日志输出
        metric_logger.info(
            f"[Batch-Log] [{decision_msg}] "
            f"Bud:{old_budget:.0f}->{target_budget} | "
            f"Sat:{saturation_rate:.1%} | "
            f"ITL:{p95_itl:.1f}ms | "
            f"TTFT:{p95_ttft:.1f}ms | "
            f"QTime:{avg_queue_latency:.0f}ms | "
            f"InTok:{total_input_tokens} | "
            f"GenTok:{total_generated_tokens} | "
            f"Err:{error_count} | "
            f"Size:{len(batch_items)} | "
            f"Dur:{batch_duration * 1000:.0f}ms"
        )


# ==========================================
# 6. Web Server,生命周期管理: 启动 App 时同时启动两个后台循环
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

        # [修改] 入队时记录到达时间 (time.time())
        arrival_time = time.time()
        await request_queue.put((req_id, prompt, token_count, arrival_time))

        return await future
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # 添加 access_log=False 参数
    uvicorn.run(app, host="0.0.0.0", port=8080, access_log=False)