# 把10个实例全部划分为交互式实例
import sys
import os
import time
import math
import logging
import redis
import statistics
from typing import List, Dict, Tuple
from enum import Enum
from kubernetes import client, config

# ==========================================
# 1. 配置与常量
# ==========================================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# RDP 策略参数
# [交互式平面]
INTERACTIVE_HEADROOM = 1.2  # alpha: 安全缓冲系数 (预留20% Budget)
EMA_BETA = 0.8  # beta: 平滑系数
DEFAULT_BUDGET = 1000  # 初始 Budget

# [批处理平面]
SLO_TTFT_MS = 200.0  # 批处理死线 (200毫秒)
BATCH_SCALING_STEP_MILD = 1  # 温和扩容步长
BATCH_SCALING_STEP_PANIC = 2  # 恐慌扩容步长

# 日志配置
LOG_DIR = "/home/ke/work2/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("RDP_Global_Scaler")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "global-scaler.log"), encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - [RDP-Scaler] - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(logging.Formatter('%(asctime)s - [RDP-Scaler] - %(message)s'))
logger.addHandler(console_handler)

logger.info(f"Global Scaler (Slim) 日志已启动，写入路径: {LOG_DIR}")


# ==========================================
# 2. 数据结构定义
# ==========================================
class Role(Enum):
    INTERACTIVE = "ROLE_INTERACTIVE"
    BATCH = "ROLE_BATCH"


class InstanceState:
    """代表一个运行中的 Pod 实例状态"""

    def __init__(self, pod_name: str, instance_id: int):
        self.pod_name = pod_name
        self.instance_id = instance_id
        self.role = Role.BATCH

        # RDP 核心指标
        self.budget = DEFAULT_BUDGET
        self.running_tokens = 0
        self.queued_tokens = 0
        self.hol_wait_ms = 0.0
        self.queue_len = 0
        self.p95_ttft = 0.0


class WarmPoolManager:
    """
    预热池管理器：纯逻辑管理，操作 Redis 标签。
    """

    def __init__(self, redis_client):
        self.redis = redis_client
        # 静态池定义
        self.interactive_pool = []  # 01-10
        self.batch_pool = [f"qwen-instance-{i:02d}" for i in range(1, 11)]

        # 内存中记录当前激活的实例 (用于快速查找空闲 Pod)
        self.active_interactive = set()
        self.active_batch = set()

    def get_active_pod_map(self) -> Dict[str, int]:
        """返回所有被激活的 Pod 映射 {name: id}"""
        active_map = {}
        for name in self.active_interactive | self.active_batch:
            try:
                iid = int(name.split("-")[-1])
                active_map[name] = iid
            except:
                pass
        return active_map

    def deploy_instance(self, role: Role) -> bool:
        """
        [逻辑扩容] 从池子里找一个 Standby 的，打上 Redis 标签
        """
        pool = self.interactive_pool if role == Role.INTERACTIVE else self.batch_pool
        active_set = self.active_interactive if role == Role.INTERACTIVE else self.active_batch

        # 1. 找一个不在激活列表里的 Pod
        candidate = None
        for pod_name in pool:
            if pod_name not in active_set:
                candidate = pod_name
                break

        if not candidate:
            logger.warning(f"[{role.value}] 预热池已耗尽！无法扩容 (Max 10)")
            return False

        logger.info(f"正在激活(WarmUp) [{role.value}]: {candidate}")

        # 2. 写入 Redis -> 流量开始进入
        self.redis.set(f"pod:{candidate}:role", role.value)

        # 3. 更新内存状态
        active_set.add(candidate)
        return True

    def delete_instance(self, instance_id: int) -> bool:
        """
        [逻辑缩容] 删除 Redis 标签，流量停止
        """
        pod_name = f"qwen-instance-{instance_id:02d}"

        if pod_name in self.active_interactive:
            self.active_interactive.remove(pod_name)
        elif pod_name in self.active_batch:
            self.active_batch.remove(pod_name)
        else:
            logger.warning(f"试图缩容未激活的实例: {pod_name}")
            return False

        logger.info(f"正在待命(Standby): {pod_name}")

        self.redis.delete(f"pod:{pod_name}:role")
        # 清理 Sidecar 指标
        self.redis.delete(f"pod:{pod_name}:budget")
        self.redis.delete(f"pod:{pod_name}:running_tokens")
        self.redis.delete(f"pod:{pod_name}:p95_ttft")
        return True


# ==========================================
# 3. RDP 全局自动伸缩器 (核心逻辑)
# ==========================================
class ReactiveDualPlaneScaler:
    def __init__(self):
        try:
            self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            self.redis.ping()
        except Exception as e:
            logger.error(f"Redis 连接失败: {e}")
            sys.exit(1)

        self.k8s = WarmPoolManager(self.redis)
        self.instances: Dict[str, InstanceState] = {}

        # [RDP Model] 平滑后的单机容量 Omega
        self.smoothed_omega = float(DEFAULT_BUDGET)

        # 冷却控制
        self.last_interactive_scale_time = 0
        self.last_batch_scale_time = 0
        # [新增] 批处理稳定性计数器 (用于去抖动)
        self.batch_zone_counter = 0
        self.batch_last_zone = None

    def wait_for_physical_pool_ready(self):
        """
        启动时阻塞检查：确保 10 个物理 Pod 确实处于 Running 状态
        """
        logger.info(">>> 正在检查物理预热池状态 (Waiting for 10 Pods)...")
        try:
            config.load_kube_config()
        except:
            config.load_incluster_config()
        v1 = client.CoreV1Api()

        expected_names = set(self.k8s.interactive_pool + self.k8s.batch_pool)

        while True:
            try:
                # 注意：这里 label_selector 必须匹配 init_warm_pool.py 中的 component=predictor
                ret = v1.list_namespaced_pod(namespace="like", label_selector="component=predictor")

                ready_names = []
                for pod in ret.items:
                    pod_name = pod.metadata.name
                    if "-predictor-" not in pod_name: continue
                    logical_name = pod_name.split("-predictor-")[0]

                    if logical_name in expected_names:
                        # 检查 2/2 Ready
                        if pod.status.phase == "Running" and pod.status.container_statuses:
                            if all(c.ready for c in pod.status.container_statuses):
                                ready_names.append(logical_name)

                missing = expected_names - set(ready_names)
                if not missing:
                    logger.info(f">>> ✅ 物理预热池就绪。")
                    break
                else:
                    logger.info(f"等待 Pod 就绪... 缺: {sorted(list(missing))}")
                    time.sleep(5)
            except Exception as e:
                logger.error(f"K8s API Error: {e}")
                time.sleep(5)

    def sync_state(self):
        """从 Redis 同步最新状态"""
        # 1. 也是最重要的一步：从 Manager 获取当前"逻辑上"激活的 Pod
        active_map = self.k8s.get_active_pod_map()
        current_names = set(active_map.keys())

        # 2. 更新内存对象
        for name in current_names:
            if name not in self.instances:
                self.instances[name] = InstanceState(name, active_map[name])
                # 读取角色
                r_str = self.redis.get(f"pod:{name}:role")
                if r_str: self.instances[name].role = Role(r_str)

            # 3. 读取 Sidecar 指标
            ins = self.instances[name]
            try:
                pipe = self.redis.pipeline()
                keys = ["budget", "running_tokens", "queued_tokens", "hol_wait_ms", "queue_len", "p95_ttft"]
                for k in keys: pipe.get(f"pod:{name}:{k}")
                res = pipe.execute()

                if res[0]: ins.budget = float(res[0])
                if res[1]: ins.running_tokens = int(res[1])
                if res[2]: ins.queued_tokens = int(res[2])
                if res[3]: ins.hol_wait_ms = float(res[3])
                if res[4]: ins.queue_len = int(res[4])
                if res[5]: ins.p95_ttft = float(res[5])
            except:
                pass

        # 4. 清理
        for name in list(self.instances.keys()):
            if name not in current_names:
                del self.instances[name]

    def update_capacity_model(self):
        """更新 EMA 平滑容量"""
        i_pods = [i for i in self.instances.values() if i.role == Role.INTERACTIVE]
        if not i_pods: return

        current_avg = statistics.mean([i.budget for i in i_pods])
        self.smoothed_omega = (EMA_BETA * self.smoothed_omega) + ((1 - EMA_BETA) * current_avg)

        logger.info(f"[Model] Smoothed Omega: {self.smoothed_omega:.0f} (Inst Avg: {current_avg:.0f})")


    # ==========================================
    # 平面 2: 批处理伸缩器
    # ==========================================
    def run_batch_scaler(self):
        b_pods = [i for i in self.instances.values() if i.role == Role.BATCH]
        curr_replicas = len(b_pods)

        # ------------------------------------------------------
        # 1. 计算核心指标 (保持原逻辑)
        # ------------------------------------------------------
        avg_ttft = 0.0
        target_pod = next((p for p in b_pods if p.instance_id == 1), None)

        if target_pod:
            avg_ttft = target_pod.p95_ttft
        else:
            avg_ttft = statistics.mean([i.p95_ttft for i in b_pods]) if b_pods else 0

        urgency = avg_ttft / SLO_TTFT_MS
        logger.info(f"[Batch] Urgency: {urgency:.1%} (AvgTTFT:{avg_ttft:.1f}ms) | Replicas: {curr_replicas}")

        # ------------------------------------------------------
        # 2. [新增] 稳定性去抖动逻辑 (Debounce)
        # ------------------------------------------------------
        # 根据 urgency 判定当前所处的“压力区间”
        current_zone = "PANIC"  # 默认最坏情况
        if urgency < 0.5:
            current_zone = "COMFORT_FAST"
        elif urgency < 0.9:
            current_zone = "COMFORT_SLOW"
        elif urgency < 1.0:
            current_zone = "WARN"
        else:
            current_zone = "PANIC"

        # 检查是否与上一次周期状态一致
        if current_zone == self.batch_last_zone:
            self.batch_zone_counter += 1
        else:
            # 状态发生突变，重置计数器
            self.batch_zone_counter = 1
            self.batch_last_zone = current_zone

        # 核心判断：如果连续次数不足 2 次，跳过本轮操作
        if self.batch_zone_counter < 2:
            logger.info(f"[Batch] ⏳ 状态未稳定 (当前:{current_zone} 计数:{self.batch_zone_counter}/2)，跳过操作")
            return

        # ------------------------------------------------------
        # 3. 执行伸缩逻辑 (原有的算法代码)
        # ------------------------------------------------------
        now = time.time()
        cooldown = 10

        # 注意：这里逻辑结构保持不变，只是被上面的 if 守卫住了
        if urgency < 0.5 and curr_replicas > 1:
            # [舒适区] -> 直接尝试缩容
            if (now - self.last_batch_scale_time) > cooldown:
                victim = sorted(b_pods, key=lambda x: x.instance_id)[-1]
                logger.info(f"[Batch] 🟢 延迟达标直接缩容: {victim.pod_name} (Urgency:{urgency:.1%})")
                self.k8s.delete_instance(victim.instance_id)
                self.last_batch_scale_time = now

        elif urgency < 0.9:
            # [舒适区] -> 检查积压
            total_queue = sum(i.queue_len for i in b_pods)
            if total_queue == 0 and curr_replicas > 1:
                if (now - self.last_batch_scale_time) > cooldown:
                    victim = sorted(b_pods, key=lambda x: x.instance_id)[-1]
                    logger.info(f"[Batch] 🟢 空闲缩容: {victim.pod_name}")
                    self.k8s.delete_instance(victim.instance_id)
                    self.last_batch_scale_time = now

        elif 0.9 <= urgency < 1.0:
            # [警戒区] -> 温和扩容
            if (now - self.last_batch_scale_time) > cooldown:
                logger.info(f"[Batch] 🟡 警戒扩容 +{BATCH_SCALING_STEP_MILD}")
                self.k8s.deploy_instance(Role.BATCH)
                self.last_batch_scale_time = now

        else:
            # [恐慌区] -> 暴力扩容
            # 这里的日志只有在稳定恐慌2个周期后才会打印
            if (now - self.last_batch_scale_time) > (cooldown / 2):
                logger.warning(f"[Batch] 🔴 恐慌扩容 +{BATCH_SCALING_STEP_PANIC}")
                for _ in range(BATCH_SCALING_STEP_PANIC):
                    success = self.k8s.deploy_instance(Role.BATCH)
                    if not success: break
                self.last_batch_scale_time = now

    def enforce_strict_initialization(self):
        """重置状态"""
        logger.info(">>> 重置预热池状态 (Redis Clean)...")
        all_pods = self.k8s.interactive_pool + self.k8s.batch_pool
        for pod in all_pods:
            self.redis.delete(f"pod:{pod}:role")

        self.k8s.active_interactive.clear()
        self.k8s.active_batch.clear()

        # 初始各激活 1 个
        self.k8s.deploy_instance(Role.BATCH)
        time.sleep(1)

    def loop(self):
        logger.info("启动全局自动伸缩器 (Slim Edition)...")
        self.wait_for_physical_pool_ready()
        self.enforce_strict_initialization()

        while True:
            try:
                self.sync_state()
                self.update_capacity_model()
                self.run_batch_scaler()
            except Exception as e:
                logger.error(f"Loop Error: {e}", exc_info=True)

            #  保持 5s 与 Sidecar 同步
            time.sleep(5)


if __name__ == "__main__":
    scaler = ReactiveDualPlaneScaler()
    scaler.loop()