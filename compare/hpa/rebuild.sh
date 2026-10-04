#!/bin/bash
# ... 配置省略 ...

# echo "🚀 [1/4] 正在删除所有 InferenceService (命名空间: like)..."
# # 1. 去掉 --wait=false，kubectl 会一直等待直到资源从集群中彻底消失
# # 2. 如果没有任何 ISVC，为了防止脚本报错退出，结尾加上 || true
# kubectl delete inferenceservice --all -n like

# echo "⏳ 确认所有 Pod 已释放..."
# # 可选：进一步确保相关 Pod 已经彻底消失
# kubectl wait --for=delete pod -l app=qwen-inference -n like --timeout=60s 2>/dev/null || true

# echo "🧹 [2/4] 清空 Redis..."
# redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" FLUSHDB

echo "📂 [2.5/4] 清理旧日志文件 (*.log)..."
rm -f /home/like/work2_final/logs/*.log
echo "   - 已清理 /home/like/work2_final/logs/ 目录下的日志。"

echo "🐳 [3/4] 正在构建镜像 (利用缓存)..."
docker build -t ke/itl-sidecar:v13 .

echo "🗑️ [4/4] 清理旧版残留 (Prune)..."
docker image prune -f

echo "✅ 搞定！现在可以安全运行部署或 TTFT 脚本了。"