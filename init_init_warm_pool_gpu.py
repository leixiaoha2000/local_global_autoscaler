import os
import subprocess
import time

NAMESPACE = os.getenv("NAMESPACE", "like")
POOL_SIZE = int(os.getenv("POOL_SIZE", "10"))
SIDECAR_IMAGE = os.getenv("SIDECAR_IMAGE", "ke/itl-sidecar:v13")
REDIS_HOST = os.getenv("REDIS_HOST", "10.154.22.10")
LOG_HOST_PATH = os.getenv("LOG_HOST_PATH", "/home/like/work2_final/logs")
GPU_MEMORY_MB = int(os.getenv("GPU_MEMORY_MB", "5000"))
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "128"))

# 定义两个 GPU UUID
GPU_UUID1 = "GPU-36fa2eff-68bc-ba6c-1f05-28a35db98afc"
GPU_UUID2 = "GPU-753adb0d-5ea3-4cc2-3430-dddfc9dcafe7"

# ================= 配置 =================
# 必须与你的 Scaler 代码中的模板完全一致
INSTANCE_YAML_TEMPLATE = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: {instance_name}
  namespace: {namespace}
  labels:                  
    app: qwen-inference
    component: predictor # 确保加上这个标签，否则 wait_for_physical_pool_ready 扫不到
  annotations:
    serving.kserve.io/enable-prometheus-scraping: "true"
    serving.kserve.io/enable-metric-aggregation: "true"
    prometheus.io/port: "8081"
    prometheus.io/path: "/metrics"
    prometheus.io/scrape: "true"
    serving.kserve.io/autoscalerClass: "external"
    nvidia.com/use-gpuuuid: "{gpu_uuid}"
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
          - --max-num-seqs={max_num_seqs}
          - --port=8081
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
            nvidia.com/gpu: 1
            nvidia.com/gpumem: {gpu_memory_mb}
          requests:
            cpu: "1"
            memory: "10Gi"
            nvidia.com/gpu: 1
            nvidia.com/gpumem: {gpu_memory_mb}

      - name: itl-sidecar
        image: {sidecar_image}
        imagePullPolicy: IfNotPresent
        ports:
          - containerPort: 8080
        env:
          - name: LOG_DIR
            value: "/mnt/logs"
          - name: REDIS_HOST
            value: "{redis_host}"
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
          path: {log_host_path}
          type: DirectoryOrCreate
"""


def create_pool():
    print(f">>> 开始初始化预热池 (qwen-instance-01 ~ {POOL_SIZE:02d})...")

    # 循环创建 POOL_SIZE 个实例
    for i in range(1, POOL_SIZE + 1):
        instance_name = f"qwen-instance-{i:02d}"
        print(f"--- 正在部署: {instance_name} ---")

        # 前5个使用 UUID1，后5个使用 UUID2（如果 POOL_SIZE > 10，可扩展）
        gpu_uuid = GPU_UUID1 if i <= 5 else GPU_UUID2

        # 1. 生成 YAML
        yaml_content = INSTANCE_YAML_TEMPLATE.format(
            instance_name=instance_name,
            namespace=NAMESPACE,
            max_num_seqs=MAX_NUM_SEQS,
            gpu_memory_mb=GPU_MEMORY_MB,
            sidecar_image=SIDECAR_IMAGE,
            redis_host=REDIS_HOST,
            log_host_path=LOG_HOST_PATH,
            gpu_uuid=gpu_uuid,          # 传入动态 UUID
        )
        filename = f"{instance_name}.yaml"

        with open(filename, "w") as f:
            f.write(yaml_content)

        # 2. 执行 kubectl apply
        try:
            subprocess.run(["kubectl", "apply", "-f", filename], check=True)
            print(f"✅ {instance_name} 已提交 (GPU UUID: {gpu_uuid})")
        except subprocess.CalledProcessError as e:
            print(f"❌ {instance_name} 部署失败: {e}")

        # 3. 清理临时文件
        if os.path.exists(filename):
            os.remove(filename)

        # 稍微间隔一下，避免对 K8s API 造成瞬间过大压力
        time.sleep(1)

    print("\n>>> 所有部署指令已下发。请等待 Pods 变为 Running 状态。")
    print(">>> 监控命令: watch -n 2 kubectl get pods -n like -l component=predictor")


if __name__ == "__main__":
    create_pool()