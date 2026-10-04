import json
import numpy as np
import os
import sys

# 尝试导入 tiktoken 以获得精确的 Token 计数
# 这与您的 Sidecar 代码逻辑保持一致
try:
    import tiktoken

    # 使用 cl100k_base 编码器 (兼容 Qwen, GPT-4 等)
    ENC = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("[Warning] 未检测到 tiktoken 库，将使用字符长度估算 (1 token ≈ 3 chars)。", file=sys.stderr)
    print("建议安装: pip install tiktoken", file=sys.stderr)

# ================= 配置 =================
# 数据集路径 (请修改为您实际的文件路径)
DATASET_PATH = "/home/ke/work2/sharegpt_prompts.json"


# =======================================

def get_token_len(text: str) -> int:
    """计算文本 Token 数"""
    if HAS_TIKTOKEN:
        try:
            return len(ENC.encode(text))
        except Exception:
            pass
    # 兜底：中文约 0.6 token/char, 英文约 0.3 token/char，这里取折中
    return max(1, len(text) // 3)


def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = DATASET_PATH

    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        print("用法: python calculate_stats.py [json文件路径]")
        return

    print(f"正在读取数据集: {file_path} ...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ JSON 解析失败: {e}")
        return

    if not isinstance(data, list):
        print("❌ 数据格式错误: 根元素应当是一个列表 (List)")
        return

    print(f"数据集加载完成，共 {len(data)} 条数据。正在计算 Token 长度...")

    token_lengths = []

    for item in data:
        # 根据您的示例，key 是 "prompt"
        # 部分 ShareGPT 数据集可能包含 "conversations" 字段，这里做兼容处理
        text = ""
        if "prompt" in item:
            text = item["prompt"]
        elif "conversations" in item:
            # 如果是多轮对话，通常取第一个 human 的提问作为 prompt 长度估算
            # 或者将所有内容拼起来。这里假设只需计算 prompt (input) 部分。
            for turn in item["conversations"]:
                if turn.get("from") == "human":
                    text += turn.get("value", "") + "\n"

        if text:
            length = get_token_len(text)
            token_lengths.append(length)

    if not token_lengths:
        print("❌ 未提取到有效的 Prompt 数据。")
        return

    # 转换为 numpy 数组进行统计计算
    arr = np.array(token_lengths)

    # 1. 计算均值 (Mean)
    mean_val = np.mean(arr)

    # 2. 计算标准差 (Std Dev)
    std_val = np.std(arr)

    # 3. 计算 P95 (95th Percentile)
    p95_val = np.percentile(arr, 95)

    print("\n" + "=" * 40)
    print("📊 数据集 Token 统计结果")
    print("=" * 40)
    print(f"样本总数 (N): {len(arr)}")
    print(f"Min: {np.min(arr)}")
    print(f"Max: {np.max(arr)}")
    print("-" * 40)
    print(f"✅ 请求长度均值 (mean):   {mean_val:.2f}")
    print(f"✅ 请求长度标准差 (std):   {std_val:.2f}")
    print(f"✅ 请求长度 P95 (p95):     {p95_val:.2f}")
    print("=" * 40)

    print("\n>>> 建议在 init_redis.py 中更新以下配置:")
    print(f'"stats:batch:mean_tokens": {int(mean_val)},')
    print(f'"stats:batch:std_tokens": {int(std_val)},')
    print(f'"stats:batch:p95_tokens": {int(p95_val)},')


if __name__ == "__main__":
    main()