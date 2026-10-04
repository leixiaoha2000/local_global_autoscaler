import redis
import os
import sys

# ================= 配置区域 =================
# Redis 连接配置 (保持与 Scaler 和 Sidecar 一致)
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# 统计学常量配置
# 这些值基于一般对话数据集 (如 ShareGPT) 的经验值
# 您可以根据实际数据集的统计分布进行调整
STATS_CONFIG = {
    # 请求长度均值 (\mu) -> 用于估算稳态下的积压工作量，这里应该改为输出长度的均值
    "stats:batch:mean_tokens": 56,

    # 请求长度标准差 (\sigma) -> 用于估算波动，这里应该改为输出长度的标准差
    "stats:batch:std_tokens": 107,

    # 请求长度 P95 分位值 (P_{95}) -> 用于小样本下的保守估算
    "stats:batch:p95_tokens": 223,

    # [可选] 系统初始吞吐量预估 (Tokens/s)
    # 虽然 Sidecar 会实时更新这个值，但在系统刚启动没有任何请求时，
    # 写入一个初始值可以防止 Scaler 计算 T_wait 时除以 0 或使用过低的默认值
    "stats:system:throughput": 50.0
}


# ===========================================

def init_redis_data():
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")

    try:
        # 建立连接
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        print("Redis connection successful.")
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        sys.exit(1)

    print("\nInitializing Statistical Constants...")
    print("-" * 40)

    # 批量写入
    pipe = r.pipeline()
    for key, value in STATS_CONFIG.items():
        pipe.set(key, value)

    # 执行并确认
    pipe.execute()

    # 验证写入结果
    for key, value in STATS_CONFIG.items():
        stored_value = r.get(key)
        print(f"  [SET] Key: {key:<25} | Value: {stored_value}")

    print("-" * 40)
    print("Initialization Complete. The Global Autoscaler is ready to read these values.")


if __name__ == "__main__":
    init_redis_data()