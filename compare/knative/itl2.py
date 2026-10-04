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
_raw_pod_name = os.getenv("POD_NAME", socket.gethostname())
if "-predictor-" in _raw_pod_name:
    POD_NAME = _raw_pod_name.split("-predictor-")[0]
else:
    POD_NAME = _raw_pod_name

print(f"[Init] Sidecar for Pod: {POD_NAME}", file=sys.stderr)

TARGET_HOST = os.getenv("TARGET_HOST", "127.0.0.1")
TARGET_PORT = os.getenv("TARGET_PORT", "8081")
METRICS_URL = f"http://{TARGET_HOST}:{TARGET_PORT}/metrics"

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))


# ==========================================
# 2. 日志配置
# ==========================================
def setup_persistent_logger():
    LOG_DIR = os.getenv("LOG_DIR", "/mnt/logs")
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except PermissionError:
        return logging.getLogger("Sidecar_Metrics")

    filename = f"sidecar-{POD_NAME}.log"
    filepath = os.path.join(LOG_DIR, filename)

    logger = logging.getLogger("Sidecar_Metrics")
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
# 4. 核心逻辑: 指标解析与计算 (完全恢复)
# ==========================================
class MetricsState:
    """用于存储上一次的Counter值，以便计算Delta"""

    def __init__(self):
        self.last_gen_tokens = -1
        # 恢复这些字段用于计算 avg_itl 和 avg_ttft
        self.last_ttft_sum = -1.0
        self.last_ttft_count = -1
        self.last_itl_sum = -1.0
        self.last_itl_count = -1
        self.last_time = time.time()


state = MetricsState()


def fetch_and_process_metrics():
    """
    抓取 vLLM 指标，计算 TPS, Avg ITL, Avg TTFT, Concurrency
    """
    try:
        response = requests.get(METRICS_URL, timeout=2)
        response.raise_for_status()

        # 初始化数据容器
        raw_data = {
            "gpu_cache": 0.0,
            "gen_tokens": 0,
            "ttft_sum": 0.0,
            "ttft_count": 0,
            "itl_sum": 0.0,
            "itl_count": 0,
            "num_running": 0,  # 新增：用于并发计算
            "num_waiting": 0  # 新增：用于并发计算
        }

        # 逐行解析
        for line in response.iter_lines():
            if not line: continue
            decoded = line.decode('utf-8')
            if decoded.startswith("#"): continue

            parts = decoded.split()
            if len(parts) < 2: continue

            key = parts[0].split('{')[0]  # 去除 label
            val = float(parts[-1])

            # --- 1. 原有性能指标 ---
            if "vllm:gpu_cache_usage_perc" in key:
                raw_data["gpu_cache"] = val
            elif key == "vllm:generation_tokens_total":
                raw_data["gen_tokens"] = val
            elif key == "vllm:time_to_first_token_seconds_sum":
                raw_data["ttft_sum"] = val
            elif key == "vllm:time_to_first_token_seconds_count":
                raw_data["ttft_count"] = int(val)
            elif key == "vllm:time_per_output_token_seconds_sum":
                raw_data["itl_sum"] = val
            elif key == "vllm:time_per_output_token_seconds_count":
                raw_data["itl_count"] = int(val)

            # --- 2. 新增并发指标 (Knative Autoscaler用) ---
            elif key == "vllm:num_requests_running":
                raw_data["num_running"] = int(val)
            elif key == "vllm:num_requests_waiting":
                raw_data["num_waiting"] = int(val)

        # --- 计算差值 (Delta) ---
        now = time.time()
        time_delta = now - state.last_time

        tps = 0.0
        avg_ttft = 0.0
        avg_itl = 0.0
        concurrency = raw_data["num_running"] + raw_data["num_waiting"]

        # 只有当不是第一次运行且有时间流逝时才计算 Delta
        if state.last_gen_tokens != -1 and time_delta > 0:
            # 1. TPS
            tokens_delta = raw_data["gen_tokens"] - state.last_gen_tokens
            if tokens_delta >= 0:
                tps = tokens_delta / time_delta

            # 2. Avg TTFT (ms)
            ttft_sum_delta = raw_data["ttft_sum"] - state.last_ttft_sum
            ttft_count_delta = raw_data["ttft_count"] - state.last_ttft_count
            if ttft_count_delta > 0:
                avg_ttft = (ttft_sum_delta / ttft_count_delta) * 1000.0

            # 3. Avg ITL (ms)
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

        return raw_data["gpu_cache"], tps, avg_itl, avg_ttft, concurrency

    except Exception as e:
        metric_logger.warning(f"Failed to fetch metrics: {e}")
        return 0, 0, 0, 0, 0


# ==========================================
# 5. 主循环
# ==========================================
def main_loop():
    metric_logger.info(f"[Start] Monitoring {METRICS_URL}...")
    INTERVAL = 5

    while True:
        try:
            # 1. 获取所有指标
            gpu_cache, tps, avg_itl, avg_ttft, concurrency = fetch_and_process_metrics()

            # 2. 日志打印 (恢复原有信息，并追加并发数)
            metric_logger.info(
                f"[Metric-Log] "
                f"GPU_Cache:{gpu_cache:.2f} | "
                f"Avg_TPS:{tps:.1f} | "
                f"Avg_ITL:{avg_itl:.1f}ms | "
                f"Avg_TTFT:{avg_ttft:.1f}ms | "
                f"Concurrency:{concurrency}"
            )

            # 3. 写入 Redis
            # (A) 写入并发数 -> 给 Autoscaler 扩缩容用
            key_conc = f"pod:{POD_NAME}:concurrency"
            redis_client.set(key_conc, concurrency, ex=15)

            # (B) 写入 GPU Cache -> 供大盘监控用 (可选)
            key_gpu = f"pod:{POD_NAME}:gpu_cache_usage"
            redis_client.set(key_gpu, gpu_cache, ex=15)

        except Exception as e:
            metric_logger.error(f"[Loop Error] {e}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main_loop()