import os
import subprocess
import time
import sys

# 尝试导入 pynvml，用于检测显存
try:
    import pynvml

    HAS_NVML = True
except ImportError:
    print("❌ 未安装 nvidia-ml-py，无法自动检测显存。请运行: pip install nvidia-ml-py")
    HAS_NVML = False

# ================= 配置 =================
# 单个实例预估占用的显存 (MB)
# 必须与 YAML 中的 nvidia.com/gpumem 保持一致
ESTIMATED_MEM_USAGE = 5000

INSTANCE_YAML_TEMPLATE = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: {instance_name}
  namespace: like
  labels:
    app: qwen-inference
    component: predictor
  annotations:
    serving.kserve.io/enable-prometheus-scraping: "true"
    serving.kserve.io/enable-metric-aggregation: "true"
    prometheus.io/port: "8081"
    prometheus.io/path: "/metrics"
    prometheus.io/scrape: "true"
    serving.kserve.io/autoscalerClass: "external"
spec:
  predictor:
    minReplicas: 1
    tolerations:
      - key: "node-role.kubernetes.io/master"
        operator: "Exists"
        effect: "NoSchedule"
      - key: "node-role.kubernetes.io/control-plane"
        operator: "Exists"
        effect: "NoSchedule"
      - key: "node.kubernetes.io/disk-pressure"
        operator: "Exists"
        effect: "NoSchedule"
    containers:
      - name: kserve-container
        image: vllm/vllm-openai:v0.6.0
        imagePullPolicy: IfNotPresent
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args:
          - --model=/mnt/models
          - --served-model-name=Qwen-0.5B
          - --max-num-seqs=256
          - --port=8081
          - --gpu-memory-utilization=0.7
          - --trust-remote-code
        env:
          - name: TRANSFORMERS_OFFLINE
            value: "1"
          - name: HF_HUB_OFFLINE
            value: "1"
        ports:
          - containerPort: 8081
            protocol: TCP
        volumeMounts:
          - name: model-volume
            mountPath: /mnt/models
        resources:
          limits:
            cpu: "2"
            memory: "16Gi"
            # 注意：如果你使用了 CUDA_VISIBLE_DEVICES 强绑，
            # 这里的 nvidia.com/gpu 可能需要设为 0 以避免 K8s 冲突，
            # 或者确保你的 K8s 允许共享设备。
            # 通常在使用 gpumem 扩展资源时，nvidia.com/gpu 可以省略或置 0
            # nvidia.com/gpu: 1
            nvidia.com/gpumem: 5000
          requests:
            cpu: "1"
            memory: "10Gi"
            # nvidia.com/gpu: 1
            nvidia.com/gpumem: 5000

      - name: itl-sidecar
        image: ke/itl-sidecar:v13
        imagePullPolicy: IfNotPresent
        ports:
          - containerPort: 8080
        env:
          - name: LOG_DIR
            value: "/mnt/logs"
          - name: REDIS_HOST
            value: "10.154.22.10"
          - name: REDIS_PORT
            value: "6379"
          - name: TARGET_HOST
            value: "127.0.0.1"
          - name: TARGET_PORT
            value: "8081"
          - name: ITL_SLO_MS
            value: "50.0"
          - name: MIN_BATCH_SIZE
            value: "1"
          - name: POD_NAME
            valueFrom:
              fieldRef:
                fieldPath: metadata.name
        volumeMounts:
          - name: log-volume
            mountPath: /mnt/logs
        resources:
          limits:
            cpu: "500m"
            memory: "512Mi"
          requests:
            cpu: "100m"
            memory: "128Mi"
    volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: qwen-0dot5b-model-pvc
      - name: log-volume
        hostPath:
          path: /home/ke/work2/logs
          type: DirectoryOrCreate
"""


def get_gpu_status():
    """
    初始化：获取所有 GPU 当前的真实剩余显存
    返回: {gpu_index: free_memory_mb}
    """
    status = {}
    if not HAS_NVML:
        # 如果没有 NVML，模拟 4 张卡，每张 24GB
        return {0: 24000, 1: 24000, 2: 24000, 3: 24000}

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb = info.free // 1024 // 1024
            status[i] = free_mb
            # print(f"GPU {i}: {free_mb} MB free")
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"⚠️ NVML 初始化失败: {e}，将使用模拟数据")
        return {0: 24000, 1: 24000, 2: 24000, 3: 24000}

    return status


def select_best_gpu(gpu_ledger):
    """
    贪心算法：选择剩余显存最多的 GPU
    """
    best_gpu = -1
    max_free = -1

    for gpu_idx, free_mem in gpu_ledger.items():
        if free_mem > max_free:
            max_free = free_mem
            best_gpu = gpu_idx

    return best_gpu


def create_pool():
    print(">>> 开始初始化预热池 (qwen-instance-01 ~ 10)...")
    print(f">>> 策略：基于本地账本的贪心分配 (单实例预估: {ESTIMATED_MEM_USAGE} MB)")

    # 1. 初始化账本 (只从硬件读取一次)
    gpu_ledger = get_gpu_status()
    print(f"--- 初始 GPU 状态: {gpu_ledger} ---")

    # 循环创建 10 个实例
    for i in range(1, 11):
        instance_name = f"qwen-instance-{i:02d}"

        # 2. 选择最佳 GPU
        target_gpu = select_best_gpu(gpu_ledger)
        current_free = gpu_ledger[target_gpu]

        # 3. 检查资源是否足够
        if current_free < ESTIMATED_MEM_USAGE:
            print(f"⚠️ 警告: GPU {target_gpu} 剩余显存 ({current_free} MB) 可能不足以部署 {instance_name}")

        print(f"--- 正在部署: {instance_name} -> GPU {target_gpu} (账本剩余: {current_free} MB) ---")

        # 4. 生成 YAML (注入 gpu_index)
        # 注意：这里我们强制设置 CUDA_VISIBLE_DEVICES
        yaml_content = INSTANCE_YAML_TEMPLATE.format(
            instance_name=instance_name,
            gpu_index=target_gpu
        )
        filename = f"{instance_name}.yaml"

        with open(filename, "w") as f:
            f.write(yaml_content)

        # 5. 执行 kubectl apply
        try:
            subprocess.run(["kubectl", "apply", "-f", filename], check=True, stdout=subprocess.DEVNULL)
            print(f"✅ {instance_name} 已提交")

            # 6. [关键] 更新本地账本 (扣除预估显存)
            # 我们不等待真实显存变化，而是直接扣除，防止后续 Pod 堆积
            gpu_ledger[target_gpu] -= ESTIMATED_MEM_USAGE

        except subprocess.CalledProcessError as e:
            print(f"❌ {instance_name} 部署失败: {e}")

        # 7. 清理临时文件
        if os.path.exists(filename):
            os.remove(filename)

        time.sleep(0.5)

    print(f"\n--- 最终 GPU 账本状态: {gpu_ledger} ---")
    print(">>> 所有部署指令已下发。请等待 Pods 变为 Running 状态。")
    print(">>> 监控命令: watch -n 2 kubectl get pods -n like -l component=predictor")


if __name__ == "__main__":
    create_pool()


