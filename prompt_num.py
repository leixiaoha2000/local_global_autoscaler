import json


def count_json_prompts(file_path):
    try:
        # 使用 utf-8 编码打开，确保能够正确读取中文内容
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # 确保根节点是列表格式
            if isinstance(data, list):
                # 统计包含 "prompt" 键的字典对象个数
                # 这种写法比简单的 len(data) 更健壮，可以过滤掉不符合格式的条目
                count = sum(1 for item in data if isinstance(item, dict) and "prompt" in item)

                print(f"统计成功！")
                print(f"文件: {file_path}")
                print(f"Prompt 总数: {count}")
            else:
                print("错误：JSON 根节点不是列表。")

    except FileNotFoundError:
        print(f"错误：未找到文件 '{file_path}'，请检查文件名是否正确。")
    except json.JSONDecodeError:
        print("错误：文件不是有效的 JSON 格式。")
    except Exception as e:
        print(f"发生未知错误: {e}")


# 指定文件名并运行
count_json_prompts('sharegpt_prompts.json')