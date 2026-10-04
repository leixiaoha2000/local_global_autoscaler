#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-like}"
REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
LOG_DIR="${LOG_DIR:-/home/like/work2_final/logs}"
SIDECAR_IMAGE="${SIDECAR_IMAGE:-ke/itl-sidecar:v13}"

echo "[1/4] Delete existing InferenceServices in namespace ${NAMESPACE}"
# kubectl delete inferenceservice --all -n "${NAMESPACE}" || true
# kubectl wait --for=delete pod -l app=qwen-inference -n "${NAMESPACE}" --timeout=120s 2>/dev/null || true

echo "[2/4] Reset Redis baseline state"
redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" FLUSHDB

echo "[3/4] Remove old experiment logs and build Llumnix compatibility sidecar"
mkdir -p "${LOG_DIR}"
find "${LOG_DIR}" -maxdepth 1 -type f -name '*.log' -delete
docker build -t "${SIDECAR_IMAGE}" .

echo "[4/4] Prune dangling image layers"
docker image prune -f
echo "Llumnix queue-adapted sidecar is ready. Return to the repository root and start init_init_warm_pool.py."

