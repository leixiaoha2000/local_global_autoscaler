import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ================= 配置 =================
LOG_FILE = "sidecar-qwen-instance-01.log"
OUTPUT_IMAGE = "p95_ttft_chart.png"


# =======================================

def parse_log(file_path):
    data = []
    # 正则匹配：时间戳 和 P95_TTFT
    # 格式示例：... P95_TTFT:63.0ms ...
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| \[Batch-Log\].*?P95_TTFT:([\d\.]+)ms")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().replace('\n[source', ' [source').replace('\n', '')

        for match in pattern.finditer(content):
            timestamp_str = match.group(1)
            val = float(match.group(2))

            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
            data.append({"time": dt, "P95_TTFT": val})

    return pd.DataFrame(data)


def plot_chart(df):
    if df.empty:
        print("未找到 P95_TTFT 数据。")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['P95_TTFT'], marker='x', markersize=4, linestyle='-', color='#d62728', label='P95 TTFT')

    plt.title('P95 TTFT Over Time (Time To First Token)', fontsize=14)
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