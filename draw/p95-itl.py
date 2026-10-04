import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ================= 配置 =================
LOG_FILE = "sidecar-qwen-instance-01.log"
OUTPUT_IMAGE = "p95_itl_chart.png"


# =======================================

def parse_log(file_path):
    data = []
    # 正则匹配：时间戳 和 P95_ITL
    # 格式示例：2026-02-01 12:19:30,927 | [Batch-Log] ... P95_ITL:12.1ms ...
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| \[Batch-Log\].*?P95_ITL:([\d\.]+)ms")

    with open(file_path, 'r', encoding='utf-8') as f:
        # 简单的多行合并逻辑（防止日志被截断）
        content = f.read().replace('\n[source', ' [source').replace('\n', '')
        # 重新按固定格式分割（假设每条日志以日期开头）
        # 这里为了简单直接用正则在全文搜索
        for match in pattern.finditer(content):
            timestamp_str = match.group(1)
            val = float(match.group(2))

            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
            data.append({"time": dt, "P95_ITL": val})

    return pd.DataFrame(data)


def plot_chart(df):
    if df.empty:
        print("未找到 P95_ITL 数据，请检查日志文件。")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['P95_ITL'], marker='o', markersize=3, linestyle='-', color='#1f77b4', label='P95 ITL')

    plt.title('P95 ITL Over Time (Inter-Token Latency)', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(OUTPUT_IMAGE)
    print(f"图表已保存至: {OUTPUT_IMAGE}")
    plt.show()


if __name__ == "__main__":
    df = parse_log(LOG_FILE)
    plot_chart(df)