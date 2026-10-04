SERVICE_URL = "http://10.101.40.218:80/openai/v1/completions"
METRICS_URL = "http://10.101.40.218:80/metrics"

import os
import requests
import json
import sys

import concurrent.futures
from tqdm import tqdm
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple


def load_prompts(file_path):
    """
    从JSON文件加载多个prompt条目，并保留索引信息
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 保留索引、原始数据和prompt内容
        prompts = [
            (idx, item, item['prompt'])
            for idx, item in enumerate(data)
            if 'prompt' in item
        ]
        return prompts
    except Exception as e:
        print(f"加载JSON文件失败: {e}", file=sys.stderr)
        sys.exit(1)


# 从 vLLM/Prometheus metrics中解析TTFT/TPOT平均值
def fetch_vllm_latency_metrics():
    """
    从 METRICS_URL 指定的 /metrics 接口中解析：
    - vllm:time_to_first_token_seconds_{sum,count}
    - vllm:time_per_output_token_seconds_{sum,count}

    返回:
    {
        "ttft_ms_avg": float or None,
        "tpot_ms_avg": float or None,
    }
    """
    metrics_url = METRICS_URL
    if not metrics_url:
        # 未配置 metrics 地址，直接跳过
        return None
    try:
        resp = requests.get(metrics_url, timeout=5)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        print(f"获取 vLLM metrics 失败: {e}", file=sys.stderr)
        return None

    def parse_avg(metric_prefix: str):
        """
        解析形如：
        vllm:time_to_first_token_seconds_sum{...} 12.345
        vllm:time_to_first_token_seconds_count{...} 100
        的多行，做 sum 后除以 count，得到平均秒数，再转毫秒。
        """
        total_sum = 0.0
        total_count = 0.0

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # sum
            if line.startswith(metric_prefix + "_sum"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        total_sum += float(parts[-1])
                    except ValueError:
                        pass

            # count
            elif line.startswith(metric_prefix + "_count"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        total_count += float(parts[-1])
                    except ValueError:
                        pass

        if total_count == 0:
            return None

        # 转换为毫秒
        return (total_sum / total_count) * 1000.0

    ttft_ms = parse_avg("vllm:time_to_first_token_seconds")
    tpot_ms = parse_avg("vllm:time_per_output_token_seconds")

    return {
        "ttft_ms_avg": ttft_ms,
        "tpot_ms_avg": tpot_ms,
    }

# Algorithm 1: 本地自动伸缩器
@dataclass
class LocalBackpressureState:
    """本地自动伸缩器的状态：历史有效吞吐量 + 上一轮的 L_eff"""
    hist_throughput_eff: float
    last_L_eff: float


class LocalBatchAutoscaler:
    """本地批处理自动伸缩器（Algorithm 1 实现）"""
    def __init__(
        self,
        w1: float,
        w2: float,
        w3: float,
        itl_slo_ms: float,
        alpha: float = 0.5,
        beta: float = 0.2,
        init_hist_throughput_eff: float = 1.0,
        init_L_eff: float = 1.0,
    ):
        if abs((w1 + w2 + w3) - 1.0) > 1e-6:
            raise ValueError("w1 + w2 + w3必须等于1")

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.itl_slo_ms = itl_slo_ms
        self.alpha = alpha
        self.beta = beta

        self.state = LocalBackpressureState(
            hist_throughput_eff=init_hist_throughput_eff,
            last_L_eff=init_L_eff,
        )

    @staticmethod
    def compute_L_eff(request_complexities: List[float]) -> float:
        """L_eff = sum_i C_{q_i}"""
        return sum(request_complexities)

    def compute_backpressure(
        self,
        actual_itl_ms: float,
        batch_processing_time_ms: float,
        last_L_eff: float,
    ) -> Tuple[float, float, float]:
        """计算LBP, TBP和LocalBP"""
        if batch_processing_time_ms <= 0:
            raise ValueError("batch_processing_time_ms 必须大于 0")

        # 延迟型背压
        lbp = actual_itl_ms / self.itl_slo_ms

        # 吞吐量型背压
        throughput_eff_curr = last_L_eff / batch_processing_time_ms
        throughput_eff_hist = self.state.hist_throughput_eff

        if throughput_eff_curr <= 0:
            tbp = float("inf")
        else:
            tbp = throughput_eff_hist / throughput_eff_curr

        local_bp = max(lbp, tbp)
        return lbp, tbp, local_bp

    def adjust_L_eff(
        self,
        local_bp: float,
        current_L_eff: float,
    ) -> float:
        """根据 Local Backpressure 调整 L_eff"""
        if local_bp < 1.0:
            amplified = (1.0 / local_bp) * current_L_eff
            new_L_eff = self.alpha * amplified + (1.0 - self.alpha) * current_L_eff
        elif local_bp > 1.0:
            new_L_eff = current_L_eff / 2.0
        else:
            new_L_eff = current_L_eff
        return new_L_eff

    @staticmethod
    def choose_batch_size(
        target_L_eff: float,
        request_complexities: List[float],
    ) -> int:
        """B = argmin_B |sum_{i=1}^B C_{q_i} - L_eff_target|"""
        if not request_complexities:
            return 0

        best_B = 1
        best_diff = float("inf")
        cumulative = 0.0

        for i, c in enumerate(request_complexities, start=1):
            cumulative += c
            diff = abs(cumulative - target_L_eff)
            if diff < best_diff:
                best_diff = diff
                best_B = i
        return best_B

    def update_and_decide_batch_size(
        self,
        actual_itl_ms: float,
        batch_processing_time_ms: float,
        last_batch_L_eff: float,
        pending_request_complexities: List[float],
    ) -> Tuple[float, int]:
        """执行一次完整的本地伸缩步骤，返回新的 L_eff 目标与建议 B"""
        lbp, tbp, local_bp = self.compute_backpressure(
            actual_itl_ms=actual_itl_ms,
            batch_processing_time_ms=batch_processing_time_ms,
            last_L_eff=last_batch_L_eff,
        )

        # 更新历史有效吞吐量（EWMA）
        throughput_eff_curr = last_batch_L_eff / batch_processing_time_ms
        self.state.hist_throughput_eff = (
            (1.0 - self.beta) * self.state.hist_throughput_eff +
            self.beta * throughput_eff_curr
        )

        # 更新当前 L_eff
        new_L_eff = self.adjust_L_eff(local_bp, last_batch_L_eff)

        # 基于剩余队列估算下一轮建议 B
        new_B = self.choose_batch_size(
            target_L_eff=new_L_eff,
            request_complexities=pending_request_complexities,
        )

        self.state.last_L_eff = new_L_eff

        return new_L_eff, new_B



# vLLM 请求发送逻辑
def send_batch(batch_items):
    """
    批量请求：将多个 prompt 一次性发送到服务端
    batch_items: [(idx, orig_data, prompt_text), ...]
    返回：与 send_request 格式相同，但为列表
    """
    service_url = SERVICE_URL
    headers = {
        "Content-Type": "application/json",
        "Host": "deepseek-r1-distill-qwen-1dot5b-model-kserve-test.example.com"
    }

    prompt_texts = [item[2] for item in batch_items]  # prompt 列表
    payload = {
        "model": "DeepSeek-R1-Distill-Qwen-1.5B",
        "prompt": prompt_texts,       # 发送数组，实现批处理
        "stream": False,
        "max_tokens": 100
    }

    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
        resp = requests.post(service_url, headers=headers, json=payload, timeout=(5, 300))
        end = time.perf_counter()
        latency = round((end - start) * 1000, 2)
        resp.raise_for_status()
        resp_json = resp.json()

        # 如果返回是 list：与每个 prompt 对应
        results = []
        for i, (idx, orig_data, prompt_text) in enumerate(batch_items):
            item_resp = resp_json[i] if isinstance(resp_json, list) else resp_json
            results.append((idx, orig_data, prompt_text, resp.status_code, item_resp, request_id, latency))
        return results

    except Exception as e:
        end = time.perf_counter()
        latency = round((end - start) * 1000, 2)
        return [
            (idx, orig_data, prompt_text, None, {"error": str(e)}, request_id, latency)
            for (idx, orig_data, prompt_text) in batch_items
        ]

# 辅助函数：请求复杂度
def approximate_token_norm_from_prompt(prompt_text: str, max_input_len: int) -> float:
    """简单近似：用字符长度近似 token 数并归一化到 [0,1]。未来可替换为真实 tokenizer。"""
    if max_input_len <= 0:
        raise ValueError("max_input_len 必须大于 0")
    token_count_approx = len(prompt_text)
    return min(token_count_approx / max_input_len, 1.0)


def build_pending_requests(prompts, max_input_len: int, w1: float, w2: float, w3: float):
    """
    从原始 prompt 列表构建待调度队列，每个元素包含：
    - idx, orig_data, prompt_text, C_q

    当前阶段：
    - C_q 仅基于 Token_norm(q) 计算（w1=1, w2=w3=0），
      Step_norm(q) / Output_norm(q) 留待未来通过预测模型补齐。
    """
    pending = []
    for idx, orig_data, prompt_text in prompts:
        token_norm = approximate_token_norm_from_prompt(prompt_text, max_input_len)

        # 占位变量：Step_norm(q), Output_norm(q)
        step_norm = None  # TODO: 未来可由历史推理步数统计或预测模块给出
        output_norm = None  # TODO: 未来可由输出 token 预测模块给出

        # 当前实现：只看 token_norm，w1=1, w2=w3=0
        C_q = w1 * token_norm

        pending.append({
            "idx": idx,
            "orig_data": orig_data,
            "prompt_text": prompt_text,
            "C_q": C_q,
            "token_norm": token_norm,
            "step_norm": step_norm,
            "output_norm": output_norm,
        })
    return pending

# 多轮自适应实验主入口

def main():
# 1.从 prompt 列表输入
# 2.根据请求复杂度进行动态批处理调度
# 3.调用 vLLM 批量推理
# 4.动态调整批大小 B
# 5.输出完整实验结果
    if len(sys.argv) < 2:
        print("使用方法: python parallel_api_requests.py <输入JSON文件> [输出JSON文件]", file=sys.stderr)
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'results.json'

    prompts = load_prompts(json_file)
    total = len(prompts)

    # 参数与本地伸缩器初始化
    MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", 2048))
    # 当前阶段仅按 Token_norm 调整复杂度，w1=1, w2=w3=0
    w1, w2, w3 = 1.0, 0.0, 0.0

    # ITL SLO (ms)，可以理解为 TPOT SLO 或 TTFT SLO
    ITL_SLO_MS = float(os.getenv("ITL_SLO_MS", "50"))
    ALPHA = float(os.getenv("ALPHA", "0.5"))
    BETA = float(os.getenv("BETA", "0.2"))

    # 构建待调度请求队列（带 C_q）
    pending = build_pending_requests(prompts, MAX_INPUT_LEN, w1, w2, w3)

    if not pending:
        print("没有有效的 prompt 条目", file=sys.stderr)
        sys.exit(0)

    # 初始 L_eff 由初始 batch_size 和平均 C_q 估计
    INIT_BATCH_SIZE = int(os.getenv("INIT_BATCH_SIZE", "4"))
    avg_C_q = sum(item["C_q"] for item in pending) / len(pending)
    init_L_eff = max(avg_C_q * INIT_BATCH_SIZE, 1e-3)

    autoscaler = LocalBatchAutoscaler(
        w1=w1,
        w2=w2,
        w3=w3,
        itl_slo_ms=ITL_SLO_MS,
        alpha=ALPHA,
        beta=BETA,
        init_hist_throughput_eff=1.0,
        init_L_eff=init_L_eff,
    )

    current_L_eff_target = init_L_eff

    print(f"总请求数: {total}", file=sys.stderr)
    print(f"初始平均 C_q: {avg_C_q:.4f}, 初始 L_eff: {init_L_eff:.4f}", file=sys.stderr)

    # 多轮自适应批处理实验主循环
    total_start_time = time.perf_counter()
    results = [None] * total
    requests_data = []

    batch_index = 0

    while pending:
        batch_index += 1

        # 根据当前L_eff_target从队列头部选择 B
        pending_C_list = [item["C_q"] for item in pending]
        B = autoscaler.choose_batch_size(current_L_eff_target, pending_C_list)
        # B不能为0，且不能超过剩余请求数
        B = max(1, min(B, len(pending)))

        current_batch = pending[:B]
        batch_items = [
            (item["idx"], item["orig_data"], item["prompt_text"])
            for item in current_batch
        ]
        batch_L_eff = sum(item["C_q"] for item in current_batch)

        # 发送当前批次并测量批次处理时间
        batch_start = time.perf_counter()
        batch_results = send_batch(batch_items)
        batch_end = time.perf_counter()
        batch_time_ms = (batch_end - batch_start) * 1000.0

        # 将结果写入全局数组
        for res in batch_results:
            idx, orig_data, prompt_text, status, result, request_id, latency = res

            # 查回该请求的 C_q
            C_q = None
            for it in current_batch:
                if it["idx"] == idx:
                    C_q = it["C_q"]
                    break

            results[idx] = {
                "request_id": request_id,
                "index": idx,
                "prompt_text": prompt_text,
                "status": status,
                "response": result,
                "latency_ms": latency,
                "C_q": C_q,
                "batch_index": batch_index,
            }

            requests_data.append({
                "request_id": request_id,
                "index": idx,
                "prompt": orig_data,
                "response": result,
                "status": status,
                "latency_ms": latency,
                "C_q": C_q,
                "batch_index": batch_index,
            })

        # 从队列中移除已处理的请求
        pending = pending[B:]

        # 获取当前 vLLM metrics（TTFT / TPOT），用 TPOT 近似 actual ITL
        metrics = fetch_vllm_latency_metrics()
        if metrics and metrics.get("tpot_ms_avg") is not None:
            actual_itl_ms = metrics["tpot_ms_avg"]
        elif metrics and metrics.get("ttft_ms_avg") is not None:
            actual_itl_ms = metrics["ttft_ms_avg"]
        else:
            # 回退：用当前批次的 HTTP 延迟近似
            if batch_results:
                # 取第一条的 latency 作为代表
                actual_itl_ms = batch_results[0][-1]
            else:
                actual_itl_ms = batch_time_ms / max(B, 1)

        # 计算下一轮的 L_eff 目标与建议 B（基于剩余队列）
        next_pending_C = [item["C_q"] for item in pending]
        new_L_eff, suggested_B = autoscaler.update_and_decide_batch_size(
            actual_itl_ms=actual_itl_ms,
            batch_processing_time_ms=batch_time_ms,
            last_batch_L_eff=batch_L_eff,
            pending_request_complexities=next_pending_C,
        )

        current_L_eff_target = new_L_eff

        # 打印本轮实验的伸缩信息
        print(
            f"[Batch {batch_index}] size={B}, L_eff={batch_L_eff:.4f}, "
            f"batch_time={batch_time_ms:.2f} ms, actual_ITL={actual_itl_ms:.2f} ms, "
            f"next_L_eff_target={new_L_eff:.4f}, suggested_next_B={suggested_B}",
            file=sys.stderr,
        )


    # 实验整体统计
    total_end_time = time.perf_counter()
    total_time = round(total_end_time - total_start_time, 2)

    success_count = sum(1 for item in results if item["status"] and item["status"] // 100 == 2)
    error_count = total - success_count
    avg_latency = round(sum(item["latency_ms"] for item in results) / total, 2)

    # 保存完整结果到 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(requests_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70, file=sys.stdout)
    print("所有请求处理完成! 自适应批处理实验统计:", file=sys.stdout)
    print("=" * 70, file=sys.stdout)
    print(f"总请求数量: {total}", file=sys.stdout)
    print(f"成功请求: {success_count}", file=sys.stdout)
    print(f"失败请求: {error_count}", file=sys.stdout)
    print(f"总处理时间: {total_time:.2f}秒", file=sys.stdout)
    print(f"平均端到端延迟(HTTP): {avg_latency:.2f}毫秒", file=sys.stdout)

    if error_count > 0:
        print("\n" + "错误请求详情:", file=sys.stdout)
        for item in results:
            if not item["status"] or item["status"] // 100 != 2:
                print(f"请求ID: {item['request_id']}", file=sys.stdout)
                print(f"索引: {item['index'] + 1}/{total}", file=sys.stdout)
                print(f"提示: {item['prompt_text']}", file=sys.stdout)
                if 'error' in item['response']:
                    print(f"错误信息: {item['response']['error']}", file=sys.stdout)
                print("-" * 50, file=sys.stdout)

    print("\n完整结果已保存到文件: " + output_file, file=sys.stdout)

if __name__ == "__main__":
    main()