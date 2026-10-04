import json

input_file = "/home/ke/work2/sharegpt/computer_zh_26k.jsonl"
output_file = "prompt.json"

max_prompts = 589  # 设置希望提取的 prompt 数量

results = []
count = 0

# 逐行读取 JSONL
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if count >= max_prompts:
            break  # 达到上限就停止读取

        if not line.strip():
            continue  # 跳过空行
        obj = json.loads(line)

        # 提取 conversation 数组中的 human
        if "conversation" in obj:
            for item in obj["conversation"]:
                if count >= max_prompts:
                    break  # 子循环也要判断上限

                if "human" in item:
                    results.append({"prompt": item["human"]})
                    count += 1

# 写入新的 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"提取完成，共提取 prompt 数量：{len(results)}（目标 {max_prompts}）")