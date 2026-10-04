SERVICE_URL = "http://10.96.97.80/openai/v1/completions"
METRICS_URL = "http://10.96.97.80/metrics"

import os
import requests
import json
import sys
import concurrent.futures
from tqdm import tqdm
import time
import uuid

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


#从vLLM/Prometheus metrics中解析TTFT/TPOT平均值
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
        的多行，做 sum 后除以 count，得到平均秒数，再转毫秒
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

# 将多个 prompt 一次性发送到服务端
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



def main():
    # 通过命令行参数指定JSON文件路径
    if len(sys.argv) < 2:
        print("使用方法: python parallel_api_requests.py <输入JSON文件> [输出JSON文件]", file=sys.stderr)
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'results.json'

    prompts = load_prompts(json_file)
    total = len(prompts)
    # 批处理大小
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))

    batches = [prompts[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    batch_total = len(batches)
    print(f"共 {batch_total} 个批次，每批最多 {BATCH_SIZE} 条请求", file=sys.stderr)

    # 初始化统计信息
    total_start_time = time.perf_counter()
    requests_data = []

    print(f"开始并行处理 {total} 个prompt请求...", file=sys.stderr)
    results = [None] * total  # 预分配结果列表

    # 使用线程池并行处理请求，最多30个请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(30, batch_total)) as executor:
        futures = [executor.submit(send_batch, batch) for batch in batches]

        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=batch_total, desc="批次进度", unit="batch",
                           file=sys.stdout):
            batch_results = future.result()

            for res in batch_results:
                idx, orig_data, prompt_text, status, result, request_id, latency = res

                results[idx] = {
                    "request_id": request_id,
                    "index": idx,
                    "prompt_text": prompt_text,
                    "status": status,
                    "response": result,
                    "latency_ms": latency
                }

                requests_data.append({
                    "request_id": request_id,
                    "index": idx,
                    "prompt": orig_data,
                    "response": result,
                    "status": status,
                    "latency_ms": latency
                })

    # 计算总体耗时
    total_end_time = time.perf_counter()
    total_time = round(total_end_time - total_start_time, 2)

    # 准备统计摘要
    success_count = sum(1 for item in results if item["status"] and item["status"] // 100 == 2)
    error_count = total - success_count
    avg_latency = round(sum(item["latency_ms"] for item in results) / total, 2)

    # 在所有请求跑完之后，抓取 vLLM 侧的TTFT/TPOT
    vllm_metrics = fetch_vllm_latency_metrics()

    # 保存完整结果到JSON文件（保持原结构不变）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(requests_data, f, ensure_ascii=False, indent=2)

    # 输出统计信息
    print("\n" + "=" * 70, file=sys.stdout)
    print("所有请求处理完成! 请求统计:", file=sys.stdout)
    print("=" * 70, file=sys.stdout)
    print(f"总请求数量: {total}", file=sys.stdout)
    print(f"成功请求: {success_count}", file=sys.stdout)
    print(f"失败请求: {error_count}", file=sys.stdout)
    print(f"总处理时间: {total_time:.2f}秒", file=sys.stdout)
    print(f"平均端到端延迟(HTTP): {avg_latency:.2f}毫秒", file=sys.stdout)

    # 输出 vLLM metrics 中的首 token / token 间延迟
    if vllm_metrics:
        ttft = vllm_metrics.get("ttft_ms_avg")
        tpot = vllm_metrics.get("tpot_ms_avg")
        print("-" * 70, file=sys.stdout)
        print("vLLM 指标统计 (通过 /metrics 解析):", file=sys.stdout)
        if ttft is not None:
            print(f"平均首 token 延迟 (TTFT): {ttft:.2f} 毫秒", file=sys.stdout)
        else:
            print("平均首 token 延迟 (TTFT): 无法从 metrics 解析", file=sys.stdout)
        if tpot is not None:
            print(f"平均 token 间延迟 (TPOT): {tpot:.4f} 毫秒", file=sys.stdout)
        else:
            print("平均 token 间延迟 (TPOT): 无法从 metrics 解析", file=sys.stdout)
    else:
        print("-" * 70, file=sys.stdout)
        print("未配置 METRICS_URL 或获取 vLLM metrics 失败，无法输出 TTFT/TPOT。", file=sys.stdout)

    # 如果有错误，打印详细信息
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

    # 输出所有结果到控制台
    print("\n完整结果已保存到文件: " + output_file, file=sys.stdout)

if __name__ == "__main__":
    main()