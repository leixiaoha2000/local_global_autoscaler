import sys
import os
import time
import socket
import requests
import redis
import logging
from logging.handlers import RotatingFileHandler

# ==========================================
# 1. 初始化配置与 Pod 名称
# ==========================================

# --- POD 名称提取 ---
_raw_pod_name = os.getenv("POD_NAME", socket.gethostname())
if "-predictor-" in _raw_pod_name:
    POD_NAME = _raw_pod_name.split("-predictor-")[0]
else:
    POD_NAME = _raw_pod_name

print(f"[Init] Sidecar for Pod: {POD_NAME}", file=sys.stderr)

# --- 目标 vLLM 地址 ---
TARGET_HOST = os.getenv("TARGET_HOST", "127.0.0.1")
TARGET_PORT = os.getenv("TARGET_PORT", "8081")
METRICS_URL = f"http://{TARGET_HOST}:{TARGET_PORT}/metrics"

# --- Redis 配置 ---
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))


# ==========================================
# 2. 日志配置 (复用右侧代码逻辑)
# ==========================================
def setup_persistent_logger():
    # 1. 确定日志挂载
    LOG_DIR = os.getenv("LOG_DIR", "/mnt/logs")
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except PermissionError:
        print(f"[Error] No permission to create {LOG_DIR}, fallback to stderr", file=sys.stderr)
        return logging.getLogger("Sidecar_Metrics")

    # 2. 构造唯一文件名字
    filename = f"sidecar-{POD_NAME}.log"
    filepath = os.path.join(LOG_DIR, filename)

    logger = logging.getLogger("Sidecar_Metrics")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    try:
        file_handler = RotatingFileHandler(
            filepath,
            mode='a',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=1,
            encoding='utf-8'
        )
        file_fmt = logging.Formatter('%(asctime)s | %(message)s')
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)

        print(f"[Init] Logging to file: {filepath}", file=sys.stderr)
    except Exception as e:
        print(f"[Error] Failed to setup logger: {e}", file=sys.stderr)

    return logger


metric_logger = setup_persistent_logger()

# ==========================================
# 3. Redis 连接
# ==========================================
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    metric_logger.info(f"[Init] Redis connected at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    metric_logger.error(f"[Fatal] Redis connection failed: {e}")
    sys.exit(1)


# ==========================================
# 4. 核心逻辑: 指标解析与计算
# ==========================================
class MetricsState:
    """用于存储上一次的Counter值，以便计算Delta"""

    def __init__(self):
        self.last_gen_tokens = -1
        self.last_ttft_sum = -1.0
        self.last_ttft_count = -1
        self.last_itl_sum = -1.0
        self.last_itl_count = -1
        self.last_time = time.time()


state = MetricsState()


def fetch_and_process_metrics():
    """
    抓取 vLLM 指标，计算 TPS, Avg ITL, Avg TTFT, GPU Cache
    返回: (gpu_cache, tps, avg_itl, avg_ttft)
    """
    try:
        response = requests.get(METRICS_URL, timeout=2)
        response.raise_for_status()

        # 临时存储解析到的原始值
        raw_data = {
            "gpu_cache": None,
            "gen_tokens": 0,
            "ttft_sum": 0.0,
            "ttft_count": 0,
            "itl_sum": 0.0,
            "itl_count": 0
        }

        # 逐行解析 Prometheus 格式
        # for line in response.iter_lines():
        #     if not line: continue
        #     decoded = line.decode('utf-8')
        #     if decoded.startswith("#"): continue
        #
        #     parts = decoded.split()
        #     if len(parts) < 2: continue
        #
        #     key = parts[0]
        #     val = float(parts[-1])
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode('utf-8')
            if decoded.startswith("#"):
                continue
            parts = decoded.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            # 关键修复：移除标签部分，只保留指标名称
            key = key.split('{')[0]  # 添加这一行
            val = float(parts[-1])

            # 1. GPU Cache
            if "vllm:gpu_cache_usage_perc" in key:
                raw_data["gpu_cache"] = val

            # 2. Token 生成总数 (Counter) -> 用于计算 TPS
            elif key == "vllm:generation_tokens_total":
                raw_data["gen_tokens"] = val

            # 3. TTFT (Sum & Count) -> 用于计算 Avg TTFT
            elif key == "vllm:time_to_first_token_seconds_sum":
                raw_data["ttft_sum"] = val
            elif key == "vllm:time_to_first_token_seconds_count":
                raw_data["ttft_count"] = int(val)

            # 4. ITL (Sum & Count) -> 用于计算 Avg ITL
            # vLLM 通常用 time_per_output_token_seconds 表示 ITL
            elif key == "vllm:time_per_output_token_seconds_sum":
                raw_data["itl_sum"] = val
            elif key == "vllm:time_per_output_token_seconds_count":
                raw_data["itl_count"] = int(val)

        # --- 计算差值 (Delta) ---
        now = time.time()
        time_delta = now - state.last_time

        # 初始化结果
        tps = 0.0
        avg_ttft = 0.0
        avg_itl = 0.0

        # 只有当不是第一次运行(last != -1)且时间有流逝时才计算
        if state.last_gen_tokens != -1 and time_delta > 0:
            # 1. TPS
            tokens_delta = raw_data["gen_tokens"] - state.last_gen_tokens
            if tokens_delta >= 0:
                tps = tokens_delta / time_delta

            # 2. Avg TTFT (毫秒)
            ttft_sum_delta = raw_data["ttft_sum"] - state.last_ttft_sum
            ttft_count_delta = raw_data["ttft_count"] - state.last_ttft_count
            if ttft_count_delta > 0:
                avg_ttft = (ttft_sum_delta / ttft_count_delta) * 1000.0

            # 3. Avg ITL (毫秒)
            itl_sum_delta = raw_data["itl_sum"] - state.last_itl_sum
            itl_count_delta = raw_data["itl_count"] - state.last_itl_count
            if itl_count_delta > 0:
                avg_itl = (itl_sum_delta / itl_count_delta) * 1000.0

        # 更新状态
        state.last_gen_tokens = raw_data["gen_tokens"]
        state.last_ttft_sum = raw_data["ttft_sum"]
        state.last_ttft_count = raw_data["ttft_count"]
        state.last_itl_sum = raw_data["itl_sum"]
        state.last_itl_count = raw_data["itl_count"]
        state.last_time = now

        return raw_data["gpu_cache"], tps, avg_itl, avg_ttft

    except Exception as e:
        metric_logger.warning(f"Failed to fetch metrics: {e}")
        return None, 0, 0, 0


# ==========================================
# 5. 主循环
# ==========================================
def main_loop():
    metric_logger.info(f"[Start] Monitoring {METRICS_URL}...")

    INTERVAL = 5

    while True:
        try:
            # 1. 获取并计算指标
            gpu_cache, tps, avg_itl, avg_ttft = fetch_and_process_metrics()

            # 2. 日志打印 (模仿右侧格式)
            # 注意: Sidecar 无法精确获取 P95，这里使用 Avg 代替，但对监控大盘来说趋势一致
            metric_logger.info(
                f"[Metric-Log] "
                f"GPU_Cache:{gpu_cache:.2f} | "
                f"Avg_TPS:{tps:.1f} | "
                f"Avg_ITL:{avg_itl:.1f}ms | "
                f"Avg_TTFT:{avg_ttft:.1f}ms"
            )

            # 3. 写入 Redis (HPA 核心指标)
            if gpu_cache is not None:
                key = f"pod:{POD_NAME}:gpu_cache_usage"
                pipe = redis_client.pipeline()
                pipe.set(key, gpu_cache)
                pipe.expire(key, 15)
                pipe.execute()

        except Exception as e:
            metric_logger.error(f"[Loop Error] {e}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main_loop()