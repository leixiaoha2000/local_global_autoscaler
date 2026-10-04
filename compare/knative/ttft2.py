import sys
import os
import time
import logging
import redis
import statistics
from kubernetes import client, config

# ==========================================
# 1. 配置与常量
# ==========================================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# --- Knative 策略配置 ---
# 设定 α (Alpha): 每个实例理想的并发处理数
TARGET_CONCURRENCY = 10  # <--- 请ß根据你的模型性能修改此值 (例如: 10, 50, 100)

# 阈值计算
SCALE_UP_THRESHOLD = 0.9 * TARGET_CONCURRENCY  # > 0.9α 扩容
SCALE_DOWN_THRESHOLD = 0.8 * TARGET_CONCURRENCY  # < 0.8α 缩容

COOLDOWN_SECONDS = 10  # 扩缩容冷却时间

# 日志配置
LOG_DIR = "/home/ke/work2/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("Global_Autoscaler")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "global-autoscaler.log"), encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - [KPA] - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(logging.Formatter('%(asctime)s - [KPA] - %(message)s'))
logger.addHandler(console_handler)

logger.info(f"Knative-Style Autoscaler 启动 | Target Alpha: {TARGET_CONCURRENCY}")


# ==========================================
# 2. 预热池管理器 (保持不变)
# ==========================================
class WarmPoolManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.all_instances = [f"qwen-instance-{i:02d}" for i in range(1, 11)]
        self.active_instances = []

    def get_active_count(self):
        return len(self.active_instances)

    def initialize(self):
        logger.info("正在重置 Redis 状态...")
        for pod in self.all_instances:
            self.redis.delete(f"pod:{pod}:role")

        first_instance = self.all_instances[0]
        self.active_instances = [first_instance]
        self.redis.set(f"pod:{first_instance}:role", "ROLE_BATCH")
        logger.info(f"初始化完成: 激活 {first_instance}")

    def scale_up(self) -> bool:
        current_count = len(self.active_instances)
        if current_count >= len(self.all_instances):
            logger.warning("扩容失败: 已达上限 (10)")
            return False

        next_instance = self.all_instances[current_count]
        self.active_instances.append(next_instance)
        self.redis.set(f"pod:{next_instance}:role", "ROLE_BATCH")
        logger.info(f"🟢 [扩容] +1 -> 激活 {next_instance} (Total: {len(self.active_instances)})")
        return True

    def scale_down(self) -> bool:
        if len(self.active_instances) <= 1:
            logger.warning("缩容失败: 最小保留 1")
            return False

        removed = self.active_instances.pop()
        self.redis.delete(f"pod:{removed}:role")
        logger.info(f"🔴 [缩容] -1 -> 停用 {removed} (Total: {len(self.active_instances)})")
        return True


# ==========================================
# 3. 主程序逻辑 (核心更改)
# ==========================================
class KnativeLikeScaler:
    def __init__(self):
        try:
            self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            self.redis.ping()
        except Exception as e:
            logger.error(f"Redis 连接失败: {e}")
            sys.exit(1)

        self.pool = WarmPoolManager(self.redis)
        self.last_scale_time = time.time()

    def wait_for_physical_pool_ready(self):
        """确保物理 Pod 存在 (省略部分 K8s 代码以简化展示，逻辑不变)"""
        logger.info(">>> 假设物理资源池已就绪...")
        # 实际使用时保留你原来的 wait_for_physical_pool_ready 逻辑
        pass

    def get_avg_concurrency(self) -> float:
        """
        读取所有[活跃]实例的并发数 (Running + Waiting) 并计算平均值
        ObservedMetricValue = Avg(Concurrency)
        """
        valid_values = []
        for instance_name in self.pool.active_instances:
            # 读取 Sidecar 写入的新 Key
            key = f"pod:{instance_name}:concurrency"
            val = self.redis.get(key)
            if val is not None:
                try:
                    valid_values.append(float(val))
                except ValueError:
                    pass

        # 如果读不到数据（例如刚启动），默认为 0
        if not valid_values:
            return 0.0

        return statistics.mean(valid_values)

    def loop(self):
        self.wait_for_physical_pool_ready()
        self.pool.initialize()

        logger.info(">>> 开始监控循环...")

        while True:
            try:
                current_time = time.time()

                # 1. 获取当前观测值 (Observed Metric)
                avg_concurrency = self.get_avg_concurrency()
                active_count = self.pool.get_active_count()

                # 2. 打印状态
                logger.info(
                    f"[Monitor] 副本数: {active_count} | "
                    f"平均并发: {avg_concurrency:.1f} | "
                    f"目标 α: {TARGET_CONCURRENCY}"
                )

                # 3. 检查冷却时间
                if current_time - self.last_scale_time < COOLDOWN_SECONDS:
                    time.sleep(5)  # 冷却期简单 sleep
                    continue

                # 4. 扩缩容策略 (Knative 简化版)
                # 扩容: Avg > 0.9 * Alpha
                if avg_concurrency > SCALE_UP_THRESHOLD:
                    logger.info(f"触发扩容: Avg({avg_concurrency}) > 0.9α({SCALE_UP_THRESHOLD})")
                    if self.pool.scale_up():
                        self.last_scale_time = current_time

                # 缩容: Avg < 0.8 * Alpha
                elif avg_concurrency < SCALE_DOWN_THRESHOLD:
                    logger.info(f"触发缩容: Avg({avg_concurrency}) < 0.8α({SCALE_DOWN_THRESHOLD})")
                    if self.pool.scale_down():
                        self.last_scale_time = current_time

            except Exception as e:
                logger.error(f"Loop Error: {e}")

            time.sleep(5)  # 循环间隔


if __name__ == "__main__":
    scaler = KnativeLikeScaler()
    scaler.loop()