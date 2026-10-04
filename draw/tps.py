import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ================= 配置 =================
LOG_FILE = "sidecar-qwen-instance-01.log"
OUTPUT_IMAGE = "avg_tps_chart.png"


# =======================================

def parse_log(file_path):
    data = []
    # 正则匹配：时间戳 和 Avg TPS
    # 格式示例：2026-02-01 12:17:03,484 | [Throughput-Stat] ... Avg TPS:0.0 toks/s ...
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| \[Throughput-Stat\].*?Avg TPS:([\d\.]+) toks/s")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().replace('\n[source', ' [source').replace('\n', '')

        for match in pattern.finditer(content):
            timestamp_str = match.group(1)
            val = float(match.group(2))

            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
            data.append({"time": dt, "Avg_TPS": val})

    df = pd.DataFrame(data)

    # === 关键步骤：清除启动阶段的 0 值 ===
    # 找到第一个非 0 值的索引，截取之后的数据
    if not df.empty:
        non_zero_indices = df[df['Avg_TPS'] > 0].index
        if not non_zero_indices.empty:
            first_valid_idx = non_zero_indices[0]
            # 也可以保留前一个 0 点作为起跳点，这里选择直接截取非0段
            df = df.loc[first_valid_idx:].copy()
        else:
            # 如果全是0，则返回空或保持原样
            print("警告：数据中未发现有效的 TPS 负载。")

    return df


def plot_chart(df):
    if df.empty:
        print("无有效 TPS 数据可绘制。")
        return

    plt.figure(figsize=(12, 6))
    # 使用面积图或者线图
    plt.plot(df['time'], df['Avg_TPS'], color='#2ca02c', linewidth=2, label='Avg TPS')
    plt.fill_between(df['time'], df['Avg_TPS'], color='#2ca02c', alpha=0.1)

    plt.title('Average TPS Over Time (Throughput)', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Tokens / sec', fontsize=12)
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