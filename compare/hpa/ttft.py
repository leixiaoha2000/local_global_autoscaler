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

# HPA 阈值配置
GPU_CACHE_THRESHOLD_UP = 0.02  # 缓存占用 > 85% -> 扩容
GPU_CACHE_THRESHOLD_DOWN = 0.01  # 缓存占用 < 60% -> 缩容
COOLDOWN_SECONDS = 10  # 扩缩容冷却时间

# 日志配置
LOG_DIR = "/home/ke/work2/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("Global_HPA")
logger.setLevel(logging.INFO)
logger.handlers.clear()

# 双流日志：文件 + 控制台
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "global-hpa.log"), encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - [HPA] - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(logging.Formatter('%(asctime)s - [HPA] - %(message)s'))
logger.addHandler(console_handler)

logger.info(f"Global HPA Scaler 启动，日志路径: {LOG_DIR}")


# ==========================================
# 2. 预热池管理器 (修复版：同步 Redis 状态)
# ==========================================
class WarmPoolManager:
    """
    管理 10 个物理 Pod 的逻辑状态。
    关键修复：在逻辑激活/停用的同时，写入 Redis 标签，供负载生成器发现。
    """

    def __init__(self, redis_client):
        self.redis = redis_client  # [新增] 需要持有 Redis 连接
        # 定义固定的 10 个实例名称 (全部用于批处理)
        self.all_instances = [f"qwen-instance-{i:02d}" for i in range(1, 11)]
        # 当前活跃的实例列表
        self.active_instances = []

    def get_active_count(self):
        return len(self.active_instances)

    def initialize(self):
        """初始化：清除旧状态，只激活第 1 个实例"""
        # 1. 安全起见，先清除所有实例的 Redis Role 标签，防止脏数据
        logger.info("正在重置 Redis 状态...")
        for pod in self.all_instances:
            self.redis.delete(f"pod:{pod}:role")

        # 2. 激活第一个
        first_instance = self.all_instances[0]
        self.active_instances = [first_instance]

        # [新增] 写入 Redis，标记为 ROLE_BATCH
        self.redis.set(f"pod:{first_instance}:role", "ROLE_BATCH")

        logger.info(f"初始化完成: 激活 {first_instance} 并写入 Redis")

    def scale_up(self) -> bool:
        """扩容: 从池中取下一个闲置实例加入活跃列表"""
        current_count = len(self.active_instances)
        if current_count >= len(self.all_instances):
            logger.warning("扩容失败: 已达到最大实例数 (10)")
            return False

        # 取下一个
        next_instance = self.all_instances[current_count]
        self.active_instances.append(next_instance)

        # [新增] 写入 Redis
        self.redis.set(f"pod:{next_instance}:role", "ROLE_BATCH")

        logger.info(f"🟢 扩容成功: 激活 {next_instance} (当前副本数: {len(self.active_instances)})")
        return True

    def scale_down(self) -> bool:
        """缩容: 移除列表末尾的实例"""
        if len(self.active_instances) <= 1:
            logger.warning("缩容失败: 必须保留至少 1 个实例")
            return False

        removed = self.active_instances.pop()

        # [新增] 删除 Redis 标签，负载生成器将不再发送流量给它
        self.redis.delete(f"pod:{removed}:role")

        logger.info(f"🔴 缩容成功: 停用 {removed} (当前副本数: {len(self.active_instances)})")
        return True


# ==========================================
# 3. 主程序逻辑
# ==========================================
class SimpleHPAScaler:
    def __init__(self):
        # 1. 连接 Redis
        try:
            self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            self.redis.ping()
            logger.info(f"Redis 连接成功: {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.error(f"Redis 连接失败: {e}")
            sys.exit(1)

        # [修改] 将 Redis 客户端传给 Manager
        self.pool = WarmPoolManager(self.redis)
        self.last_scale_time = time.time()

    def wait_for_physical_pool_ready(self):
        """
        [保留原功能] 启动前阻塞检查：确保 K8s 中 10 个 Pod 确实都 Running
        """
        logger.info(">>> 正在检查物理集群状态 (Waiting for 10 Pods)...")
        try:
            config.load_kube_config()
        except:
            config.load_incluster_config()
        v1 = client.CoreV1Api()

        expected_names = set(self.pool.all_instances)

        while True:
            try:
                # 假设 namespace 是 'like'，label 是 'component=predictor'
                ret = v1.list_namespaced_pod(namespace="like", label_selector="component=predictor")

                ready_names = []
                for pod in ret.items:
                    # 解析 Pod 名字逻辑，适配 KServe 的命名规则
                    pod_name = pod.metadata.name
                    logical_name = pod_name
                    if "-predictor-" in pod_name:
                        logical_name = pod_name.split("-predictor-")[0]

                    if logical_name in expected_names:
                        if pod.status.phase == "Running":
                            ready_names.append(logical_name)

                missing = expected_names - set(ready_names)
                if not missing:
                    logger.info(">>> ✅ 物理资源池就绪 (10/10 Running).")
                    break
                else:
                    logger.info(f"等待 Pod 就绪... 缺: {list(missing)}")
                    time.sleep(5)
            except Exception as e:
                logger.error(f"K8s API Error: {e}")
                time.sleep(5)

    def get_avg_gpu_cache(self) -> float:
        """从 Redis 读取所有[活跃]实例的 GPU Cache 利用率并计算平均值"""
        valid_values = []

        for instance_name in self.pool.active_instances:
            # 读取 Sidecar 写入的 Key
            key = f"pod:{instance_name}:gpu_cache_usage"
            val = self.redis.get(key)

            if val is not None:
                try:
                    valid_values.append(float(val))
                except ValueError:
                    pass

        if not valid_values:
            return 0.0

        return statistics.mean(valid_values)

    def loop(self):
        # 1. 启动检查
        self.wait_for_physical_pool_ready()

        # 2. 初始化逻辑池 (Reset to 1 instance)
        self.pool.initialize()

        logger.info(">>> 开始 HPA 监控循环...")

        while True:
            try:
                current_time = time.time()

                # A. 获取指标
                avg_usage = self.get_avg_gpu_cache()
                active_count = self.pool.get_active_count()

                logger.info(f"[Monitor] 副本数: {active_count} | 平均 GPU Cache: {avg_usage:.2%}")

                # B. 检查冷却时间
                if current_time - self.last_scale_time < COOLDOWN_SECONDS:
                    time.sleep(5)
                    continue

                # C. 决策逻辑 (简单的阈值判断)
                if avg_usage > GPU_CACHE_THRESHOLD_UP:
                    # 负载过高 -> 扩容
                    if self.pool.scale_up():
                        self.last_scale_time = current_time

                elif avg_usage < GPU_CACHE_THRESHOLD_DOWN:
                    # 负载过低 -> 缩容
                    if self.pool.scale_down():
                        self.last_scale_time = current_time

            except Exception as e:
                logger.error(f"Loop Error: {e}")

            time.sleep(5)


if __name__ == "__main__":
    scaler = SimpleHPAScaler()
    scaler.loop()