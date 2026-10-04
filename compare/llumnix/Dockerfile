FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY sidecar-requirements.txt .
RUN pip install --no-cache-dir -r sidecar-requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY itl.py /app/itl_sidecar.py

ENV TARGET_HOST=127.0.0.1 \
    TARGET_PORT=8081 \
    REDIS_HOST=127.0.0.1 \
    REDIS_PORT=6379 \
    SIDECAR_BASELINE=llumnix-queue \
    INSTANCE_CAPACITY_TOKENS=8192

EXPOSE 8080
CMD ["python", "itl_sidecar.py"]

