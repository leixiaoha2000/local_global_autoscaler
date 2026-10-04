# TokenScale 与 Llumnix 场景化复现说明

## 1. 仓库与实验场景

本仓库顶层不是 Git 仓库；`llumnix-ray/` 是独立 Git 仓库，当前位于 `main`，版本依赖为 vLLM 0.6.3.post1、Ray 2.45–2.47。现有代码的主要实验栈是：

- Kubernetes/KServe 上运行 Qwen-0.5B 或 DeepSeek-R1-Distill-Qwen-1.5B；
- 通过 HAMi 的 `nvidia.com/gpumem` 在四张 GPU 上建立最多十个预热模型实例；
- 每个实例包含 vLLM 容器和一个代理/监控 sidecar；
- Redis 的 `pod:<name>:role` 将实例划分为 `ROLE_INTERACTIVE`、`ROLE_BATCH` 或 mixed；
- interactive 与 batch 使用相同模型，但有不同 TTFT/ITL SLO 和动态批大小；
- 历史基线位于 `compare/hpa`、`compare/knative`、`compare/chiron`；
- 负载来自 ShareGPT prompt 和 `shiyong/*.csv` 的两秒窗口请求数；
- 历史日志重点记录 TTFT、ITL/TPOT、输出 TPS、HOL、队列长度和活动实例数。

新基线沿用相同数据和 SLO，并统一输出逐请求 JSONL 和汇总 JSON，避免各基线使用不同统计口径。

## 2. 论文机制到当前场景的映射

### TokenScale-adapted

原论文面向 prefill/decode 分离部署，本仓库是 monolithic vLLM，并按服务等级分为 interactive/batch。因此采用以下等价映射：

| TokenScale 论文概念 | 本仓库中的实现 |
|---|---|
| Prefill token velocity | interactive 输入 token 到达率 / 单实例 SLO 内输入 token capacity |
| Decode token velocity | 按输入/预测输出长度分桶后的 batch 总 token 到达率 / 单实例实测 capacity |
| Output predictor | 使用请求 `max_tokens` 作为保守预测；离线画像仍按 S/M/L 九个桶统计 |
| Convertible Decoder | 固定的 `ROLE_MIXED` 实例，正常承担 batch，突发时最多接收一个 interactive prefill |
| SLO-aware restricted prefill | 只有 regular interactive 实例预计不能满足 TTFT 时才使用 mixed，并限制 mixed 上同时只有一个 interactive prefill |
| Token-velocity autoscaling | 根据 token rate/velocity 直接计算所需 interactive、batch、mixed 数，扩容立即、缩容带 hysteresis |

这保留了论文的两个核心创新：token-level leading indicator 和无需冷启动的可转换突发缓冲。它不声称复现 PD/KV 传输，因为当前部署没有 PD 分离。

### Llumnix-native

使用下载的 `llumnix-ray` 后端，保留：

- 基于剩余 KV capacity 的 load-aware dispatch；
- 持续的跨实例重调度；
- append-only KV cache 的多阶段 live migration；
- defragmentation 和 scale-down 前 drain。

当前 `llumnix-ray/main` 已弃用论文中的 proactive autoscaler，因此 native 路径使用固定四实例，专门评估调度和真实 KV migration。资源伸缩部分在下面的 queue 路径复现。

### Llumnix-queue

这是兼容现有 KServe sidecar 的降级模式：

- 使用论文公式 `freeness = (capacity - sum(virtual_usage)) / batch_size`；
- interactive 映射为高优先级，获得 execution headroom；
- batch 映射为普通优先级；
- head-of-line 排队请求按完整 token demand 计入 virtual usage，用于 defragmentation；
- 按最低/最高 freeness 配对 source/destination，普通、短请求优先迁移；
- 新实例自动吸收负载，缩容实例先 drain。

该模式只能迁移尚未发给 vLLM 的排队请求，输出中的 `mode_disclosure` 和 `live_kv_migration=false` 会明确标记。论文级运行中 KV cache 迁移必须使用 `Llumnix-native`，不能用 queue 结果替代。

## 3. 公平评测指标

所有新脚本使用同一流式时间戳口径并记录：

- 请求数、成功率和失败原因；
- 输入/输出/总 token 数与 TPS、成功 QPS；
- TTFT mean/P50/P95/P99；
- 每请求平均 ITL 和全部 token gap 的 mean/P50/P95/P99；
- E2E 与调度队列等待的 mean/P50/P95/P99；
- interactive、batch 分别及总体 SLO attainment；
- goodput（每秒满足对应 TTFT+ITL SLO 的请求数）；
- instance-seconds、平均活动实例数、角色切换或迁移事件。

默认 SLO 与现有脚本保持一致：interactive TTFT 200 ms、ITL 50 ms；batch TTFT 2000 ms、ITL 100 ms。若论文正文采用其他值，应在同一轮所有基线中一起修改，不能只调整新基线。

## 4. 准备实例地址

复制 `compare/tokenscale/instances.example.json`，为每个 warm-pool 实例填写 sidecar 的 OpenAI completion 地址。例如在服务器上获取 Pod IP：

```bash
kubectl get pod -n like -l component=predictor -o wide
```

实例文件中的 URL 应指向 `http://<pod-ip>:8080/v1/completions`。TokenScale 和 Llumnix-queue 必须使用完全相同的实例列表。

安装轻量评测依赖：

```bash
pip install -r compare/tokenscale/requirements.txt
```

所有命令均从仓库顶层执行，并使用模块方式 `python -m ...`。

## 5. TokenScale 离线画像

`profile.example.json` 只是格式示例，不能作为论文结果。先对每个模型/GPU 配置执行递增负载的 fixed-pool saturation sweep：

```bash
python -m compare.fixed_pool_benchmark \
  --instances compare/tokenscale/instances.json \
  --prompts sharegpt_prompts.json \
  --workload shiyong/calary2_sampled.csv \
  --output-dir results/profile-qwen05b \
  --model Qwen-0.5B \
  --window-seconds 2 \
  --workload-is-per-class \
  --interactive-ratio 0.5 \
  --max-tokens 100
```

选择仍保持目标 SLO 的最高负载运行，生成 velocity profile：

```bash
python -m compare.tokenscale.offline_profiler \
  results/profile-qwen05b/requests.jsonl \
  --output compare/tokenscale/profile.qwen05b.json \
  --ttft-slo-ms 200 \
  --itl-slo-ms 50
```

画像脚本采用 SLO-attaining 一秒 token-rate bin 的 P95。更严格的实验可分别运行九种 S/M/L 输入输出组合；脚本会自动写入相应 bucket。

## 6. 运行 TokenScale-adapted

```bash
python -m compare.tokenscale.benchmark \
  --instances compare/tokenscale/instances.json \
  --profile compare/tokenscale/profile.qwen05b.json \
  --prompts sharegpt_prompts.json \
  --workload shiyong/calary2_sampled.csv \
  --output-dir results/tokenscale \
  --model Qwen-0.5B \
  --window-seconds 2 \
  --workload-is-per-class \
  --control-interval 0.5 \
  --interactive-ratio 0.5 \
  --max-instances 10 \
  --convertible-instances 1 \
  --redis-url redis://127.0.0.1:6379/0
```

`--redis-url` 可省略；提供时会同步更新现有 `pod:<name>:role`，方便复用 sidecar 日志和可视化。

## 7. 运行 Llumnix-queue

```bash
python -m compare.llumnix.benchmark_queue \
  --instances compare/tokenscale/instances.json \
  --prompts sharegpt_prompts.json \
  --workload shiyong/calary2_sampled.csv \
  --output-dir results/llumnix-queue \
  --model Qwen-0.5B \
  --window-seconds 2 \
  --workload-is-per-class \
  --interactive-ratio 0.5 \
  --min-instances 2 \
  --max-instances 10 \
  --instance-capacity-tokens 8192 \
  --priority-headroom-tokens 1024 \
  --migrate-out-freeness 512 \
  --migrate-in-freeness 2048 \
  --scale-up-freeness 512 \
  --scale-down-freeness 4096 \
  --max-concurrency 8
```

`instance-capacity-tokens` 应由 vLLM 的可用 KV blocks 换算；8192 只是 Qwen-0.5B 小规模起点。四个 freeness 阈值必须使用同一 token/batch 单位重新校准。`priority-headroom-tokens` 应通过单请求 decode speed profile 选择，目标是 interactive 请求不出现明显 decode interference。

## 8. 运行 Llumnix-native

native 模式要求 Linux、CUDA 和独立 Python 3.9/3.10 环境。不要在当前 KServe vLLM 0.6.0 容器中直接混装；下载代码要求 vLLM 0.6.3.post1 和 Ray 2.45–2.47。

```bash
cd llumnix-ray
python -m venv .venv
source .venv/bin/activate
pip install -e '.[vllm]'

MODEL_PATH=/home/ke/model/Qwen-0.5B \
CONFIG_FILE=../compare/llumnix/native_config.yml \
bash ../compare/llumnix/launch_native.sh
```

健康检查：

```bash
curl http://127.0.0.1:1234/health
```

另开终端，从仓库顶层运行统一评测：

```bash
python -m compare.llumnix.benchmark_native \
  --endpoint http://127.0.0.1:1234/generate \
  --prompts sharegpt_prompts.json \
  --workload shiyong/calary2_sampled.csv \
  --output-dir results/llumnix-native \
  --window-seconds 2 \
  --workload-is-per-class \
  --interactive-ratio 0.5 \
  --initial-instances 4 \
  --instance-log-csv llumnix-ray/results/llumnix-native/server.log_instance.csv \
  --server-log llumnix-ray/results/llumnix-native/server.log
```

当前 official main 的 native priority 接口与 OSDI artifact 不完全一致，故 native 结果主要用于真实 migration/defragmentation；interactive/batch priority isolation 以 queue 模式评估。若论文主表需要完整优先级+KV migration，应切换论文 Appendix 指定的 `osdi24ae` artifact 分支另做一组，不能混写为当前 main 的结果。

## 9. 汇总对比

```bash
python -m compare.summarize_baselines \
  fixed=results/profile-qwen05b/summary.json \
  tokenscale=results/tokenscale/summary.json \
  llumnix_queue=results/llumnix-queue/summary.json \
  llumnix_native=results/llumnix-native/summary.json \
  --csv results/new_baselines.csv
```

每个输出目录包含：

- `requests.jsonl`：逐请求原始记录，可重新计算分位数；
- `summary.json`：统一指标和资源事件；
- `new_baselines.csv`：用于论文表格的横向汇总。

## 10. 结果解释边界

1. warm-pool 中 `average_active_instances` 是逻辑活动容量。未激活 Pod 仍占用 HAMi gpumem 时，它不等价于真实物理 GPU 成本；成本图应补充 Prometheus/DCGM GPU-seconds。
2. TokenScale 的 `profile.example.json` 不能用于正式结果，必须在实际模型、GPU、`max-num-seqs` 和显存配置下重做。
3. Llumnix-queue 的迁移事件不是 KV live migration；正式表格必须保留 `queue`/`native` 后缀。
4. 原有负载生成器只读取流前 100 bytes 且没有等待所有异步任务完成，不适合作为新基线的最终延迟统计。新脚本会完整消费流并等待所有请求结束。
5. 所有基线必须固定相同 prompt、到达序列、随机种子、生成长度、temperature、实例地址和 SLO，再重复至少三次报告均值与误差。

本仓库历史实验会同时启动 interactive 和 batch 两个生成器，并让两者各自完整重放同一 CSV；因此上述正式命令均带 `--workload-is-per-class`，总请求数是 CSV 之和的两倍。如果要把 CSV 解释为两类流量合计，则去掉该开关，并用 `--interactive-ratio` 划分，两种口径不能混在同一张结果表中。
