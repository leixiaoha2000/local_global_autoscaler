import pandas as pd

# 1. 读取原始数据集
# 由于原始文件似乎没有表头，这里设置 header=None
df = pd.read_csv('calary.csv', header=None)

# 2. 每两行取一行 (采样步长为 2)
# iloc[::2] 表示从第0行开始，每隔2行取一行
df_sampled = df.iloc[::2]

# 3. 将结果保存为新的 CSV 文件
# index=False 表示不保存索引，header=False 表示不保存表头（保持与原文件格式一致）
output_filename = 'calary_sampled.csv'
df_sampled.to_csv(output_filename, index=False, header=False)

print(f"处理完成！")
print(f"原数据行数: {len(df)}")
print(f"新数据行数: {len(df_sampled)}")
print(f"结果已保存至: {output_filename}")