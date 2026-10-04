#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/ke/model/Qwen-0.5B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-1234}"
INITIAL_INSTANCES="${INITIAL_INSTANCES:-4}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/native_config.yml}"

export HEAD_NODE_IP="${HEAD_NODE_IP:-127.0.0.1}"
export HEAD_NODE=1
export PYTHONPATH="${REPO_ROOT}/llumnix-ray${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${REPO_ROOT}/results/llumnix-native"
cd "${REPO_ROOT}"

python -m llumnix.entrypoints.vllm.api_server \
  --config-file "${CONFIG_FILE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --initial-instances "${INITIAL_INSTANCES}" \
  --launch-ray-cluster \
  --model "${MODEL_PATH}" \
  --worker-use-ray \
  --migration-backend rayrpc \
  --enable-routine-migration \
  --enable-pre-stop-migration \
  --enable-defrag \
  --trust-remote-code
