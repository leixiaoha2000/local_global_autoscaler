import redis
import os
import time

# 适配你的 Redis 配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))


def check_ttft():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    print(f"--- 正在扫描 Redis ({REDIS_HOST}:{REDIS_PORT}) ---")

    # 1. 找到所有 Pod
    keys = r.keys("pod:*:role")
    if not keys:
        print("❌ 未找到任何 Pod (key: pod:*:role)")
        return

    for key in keys:
        # key 格式: pod:<name>:role
        pod_name = key.split(":")[1]

        # 2. 读取 TTFT
        ttft_key = f"pod:{pod_name}:p95_ttft"
        ttft_val = r.get(ttft_key)

        # 3. 读取更新时间 (看是否过期)
        ttl = r.ttl(ttft_key)

        print(f"Pod: {pod_name}")
        if ttft_val is None:
            print(f"   ⚠️  Redis 中没有 key: {ttft_key}")
            print(f"       -> 可能原因: Sidecar 代码没更新 / Reporter 线程崩溃")
        else:
            print(f"   ✅ Redis 值: {ttft_val} (类型: {type(ttft_val)})")
            print(f"   ⏳ TTL: {ttl} 秒")

        print("-" * 30)


if __name__ == "__main__":
    while True:
        check_ttft()
        time.sleep(2)