# 1. 基础镜像
FROM python:3.11-slim

# 2. 设置工作目录
WORKDIR /app

# 3. [优化点一] 安装系统依赖
# 通常 numpy 和 tiktoken 都有预编译的 wheel 包，不需要 gcc 编译
# 去掉 gcc 和 python3-dev 可以大幅减少构建时间和镜像体积
# 如果构建报错提示缺少编译工具，再把 gcc 加回来
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. [优化点二] 先只复制依赖文件
# 这一步非常关键！Docker 会检查 requirements.txt 内容是否变化。
# 如果没变，直接使用缓存，跳过 pip install。
COPY requirements.txt .

# 5. [优化点二] 安装 Python 依赖
# 使用阿里源或清华源可以加速下载 (可选，视网络情况而定)
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 6. [优化点三] 最后才复制源代码
# 只有当你修改了 ITL-0.5b-v13.py 时，构建过程才会从这一行开始重新执行。
# 前面的 apt 和 pip 步骤全部瞬间完成（使用缓存）。
#COPY ITL-0.5b-v25-batch.py /app/itl_sidecar.py
COPY /home/ke/work2/compare/ITL.py /app/itl_sidecar.py

# 7. 环境变量与启动 (保持不变)
ENV TARGET_HOST="127.0.0.1" \
    TARGET_PORT="8081" \
    REDIS_HOST="127.0.0.1" \
    REDIS_PORT=6379

EXPOSE 8080

CMD ["python", "itl_sidecar.py"]