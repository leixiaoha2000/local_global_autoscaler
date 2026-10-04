'''
load-v6.py:
1. 均匀到达 (Constant Arrival)
2. 读取完数据集后停止 (No Loop)
3. 双路日志 (Console + File)
'''
import asyncio
import aiohttp
import json
import csv
import time
import random
import logging
import os
import redis
from kubernetes import client, config
from typing import List

# ================= 配置区域 =================
# 1. 基础配置
NAMESPACE = "like"
LABEL_SELECTOR = "component=predictor"
SIDECAR_PORT = 8080
API_ENDPOINT = "/v1/completions"

# 2. 身份目标配置
TARGET_ROLE = os.getenv("TARGET_ROLE", "ROLE_INTERACTIVE")

# 3. Redis 配置
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# 4. 数据集配置
PROMPTS_FILE = "/home/ke/work2/sharegpt_prompts.json"
# WORKLOAD_FILE = "/home/ke/work2/shiyong/calary_sampled.csv" #批处理
WORKLOAD_FILE = "/home/ke/work2/shiyong/calary2_sampled.csv" #交互式

# 5. 时间窗口
TIME_WINDOW_DURATION = 2

# 6. 日志配置
LOG_DIR = "/home/ke/work2/logs"
LOG_FILE_NAME = "load-gen.log"


# ===========================================

# --- [新增] 双路日志配置函数 ---
def setup_logger():
    # 确保日志目录存在
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, LOG_FILE_NAME)

    # 创建 logger
    logger = logging.getLogger("LoadGen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清理旧 handler 防止重复

    # 1. 文件 Handler (写入文件)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_fmt = logging.Formatter('%(asctime)s - [Gen] - %(message)s')
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # 2. 控制台 Handler (输出到终端)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_fmt)
    logger.addHandler(console_handler)

    return logger


# 初始化全局 logger
logger = setup_logger()


class RoleAwareDiscovery:
    """角色感知发现器"""

    def __init__(self, redis_client):
        self.target_ips = []
        self.redis = redis_client
        # try:
        #     config.load_kube_config()
        # except:
        #     config.load_incluster_config()
        kube_conf_path = "/home/ke/.kube/config"

        # 针对特定环境的强力修复
        if os.path.exists(kube_conf_path):
            config.load_kube_config(config_file=kube_conf_path)
        else:
            # 兜底：如果指定文件不存在，尝试读取环境变量或默认路径
            try:
                config.load_kube_config()
            except Exception:
                # 只有在 Pod 内部运行时才用这个，你在服务器直接跑不需要这个，但也留着吧
                config.load_incluster_config()
        self.v1 = client.CoreV1Api()

    async def start_watching(self):
        while True:
            try:
                pods = self.v1.list_namespaced_pod(
                    namespace=NAMESPACE,
                    label_selector=LABEL_SELECTOR,
                    field_selector="status.phase=Running"
                )

                candidates = {}
                for p in pods.items:
                    if not p.metadata.name.startswith("qwen-instance-"):
                        continue
                    if p.status.pod_ip:
                        candidates[p.metadata.name] = p.status.pod_ip

                if not candidates:
                    self.target_ips = []
                    await asyncio.sleep(2)
                    continue

                valid_ips = []
                query_keys = []
                pod_names_list = []

                for pod_name in candidates.keys():
                    logical_name = pod_name
                    if "-predictor-" in pod_name:
                        logical_name = pod_name.split("-predictor-")[0]

                    redis_key = f"pod:{logical_name}:role"
                    query_keys.append(redis_key)
                    pod_names_list.append(pod_name)

                pipe = self.redis.pipeline()
                for k in query_keys:
                    pipe.get(k)
                roles = pipe.execute()

                for i, role_val in enumerate(roles):
                    if not role_val: continue

                    if role_val == TARGET_ROLE:
                        actual_pod_name = pod_names_list[i]
                        valid_ips.append(candidates[actual_pod_name])

                self.target_ips = sorted(list(set(valid_ips)))
            except Exception as e:
                logger.error(f"服务发现异常: {e}")

            await asyncio.sleep(2)


class Dispatcher:
    """Round-Robin 分发器"""

    def __init__(self, discovery: RoleAwareDiscovery):
        self.discovery = discovery
        self._counter = 0

    def get_next_target(self) -> str:
        ips = self.discovery.target_ips
        if not ips:
            return None
        idx = self._counter % len(ips)
        target = ips[idx]
        self._counter += 1
        return target


class UniformLoadGenerator:
    """
     均匀负载生成器
    1. 移除无限循环，读完即止
    2. 移除泊松分布，改为均匀间隔发送
    """

    def __init__(self, prompts, workload, dispatcher):
        self.prompts = prompts
        self.workload = workload  # 直接保存 list，不再 cycle
        self.dispatcher = dispatcher
        self.stats = {"success": 0, "fail": 0}

    async def send_request(self, session, idx):
        target_ip = self.dispatcher.get_next_target()
        if not target_ip:
            return

        prompt = random.choice(self.prompts)
        url = f"http://{target_ip}:{SIDECAR_PORT}{API_ENDPOINT}"

        payload = {
            "model": "Qwen-0.5B",
            "prompt": prompt,
            "max_tokens": 100,
            "stream": True,
            "temperature": 0.7
        }

        try:
            async with session.post(url, json=payload, timeout=10) as resp:
                if resp.status == 200:
                    await resp.content.read(100)
                    self.stats["success"] += 1
                else:
                    self.stats["fail"] += 1
        except Exception:
            self.stats["fail"] += 1

    async def run(self):
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            logger.info(f">>> 启动均匀负载生成 [{TARGET_ROLE}] | 窗口={TIME_WINDOW_DURATION}s")
            logger.info(f">>> 总计划窗口数: {len(self.workload)}")

            for win_idx, req_count in enumerate(self.workload, 1):
                # 1. 标记窗口开始的绝对时间戳
                window_start_time = time.perf_counter()

                num_instances = len(self.dispatcher.discovery.target_ips)

                if num_instances == 0:
                    logger.warning(f"[Win {win_idx}] 无 {TARGET_ROLE} 实例，跳过 {req_count} 请求")
                    # 即使跳过，也要保持时间窗口对齐
                    await asyncio.sleep(TIME_WINDOW_DURATION)
                    continue

                logger.info(
                    f"[Win {win_idx}/{len(self.workload)}] 目标: {req_count} | 实例: {num_instances} | 角色: {TARGET_ROLE}")

                if req_count > 0:
                    # === [修改后] 基于目标时间戳的均匀分布 ===

                    # 计算精确的理论间隔 (不再乘以 0.95，追求精准 QPS)
                    ideal_interval = TIME_WINDOW_DURATION / req_count

                    for i in range(req_count):
                        # A. 立即发射任务
                        asyncio.create_task(self.send_request(session, i))

                        # B. 计算“这一个”请求结束后的理论时间点
                        # 例如：第0个发完，理论时间应该是 start + 1*interval
                        #       第1个发完，理论时间应该是 start + 2*interval
                        target_time = window_start_time + (i + 1) * ideal_interval

                        # C. 计算由于代码执行消耗，还需要睡多久才能到达那个理论点
                        now = time.perf_counter()
                        sleep_duration = target_time - now

                        # D. 只有当确实还有剩余时间时才睡
                        # 如果 sleep_duration < 0，说明系统已经慢了，就不睡了，直接进入下一次循环追赶进度
                        if sleep_duration > 0:
                            await asyncio.sleep(sleep_duration)
                else:
                    await asyncio.sleep(TIME_WINDOW_DURATION)

                # === 窗口末尾对齐 ===
                # 防止因累积误差导致提前进入下一个窗口
                # 这里的逻辑是：确保本窗口至少耗时 TIME_WINDOW_DURATION
                elapsed = time.perf_counter() - window_start_time
                remaining = TIME_WINDOW_DURATION - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)

            logger.info(">>> 所有负载已发送完毕。任务结束。")


def load_data():
    prompts = ["Hello"]
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], str):
                        prompts = data
                    elif isinstance(data[0], dict) and 'prompt' in data[0]:
                        prompts = [x['prompt'] for x in data]
        except Exception as e:
            logger.error(f"加载 Prompts 失败: {e}")

    default_workload = [10]
    loaded_workload = []

    if os.path.exists(WORKLOAD_FILE):
        try:
            with open(WORKLOAD_FILE, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                for line_num, row in enumerate(reader, 1):
                    if not row: continue
                    val_str = row[0].strip()
                    if not val_str: continue
                    try:
                        val = int(float(val_str))
                        loaded_workload.append(val)
                    except ValueError:
                        logger.warning(f"CSV 第 {line_num} 行无法解析为数字: '{row[0]}'")
        except Exception as e:
            logger.error(f"打开 Workload CSV 失败: {e}")

    workload = loaded_workload if loaded_workload else default_workload
    logger.info(f"加载数据: Prompts={len(prompts)}, Workload_Steps={len(workload)}")
    return prompts, workload


if __name__ == "__main__":
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        logger.info(f"Redis 连接成功: {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.error(f"Redis 连接失败: {e}")
        exit(1)

    prompts, workload = load_data()

    discovery = RoleAwareDiscovery(r)
    dispatcher = Dispatcher(discovery)
    # 使用修改后的类
    gen = UniformLoadGenerator(prompts, workload, dispatcher)

    loop = asyncio.get_event_loop()
    # 启动后台发现任务
    discovery_task = loop.create_task(discovery.start_watching())

    try:
        loop.run_until_complete(gen.run())
    except KeyboardInterrupt:
        logger.info("用户手动停止。")
    finally:
        # 任务结束后，清理后台任务
        discovery_task.cancel()
        # 给一点时间让 pending tasks 完成或清理
        try:
            loop.run_until_complete(asyncio.sleep(0.5))
        except:
            pass
        logger.info("程序退出。")