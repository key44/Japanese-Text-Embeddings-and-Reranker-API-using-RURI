FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# requirements.txtをコピーして依存関係をインストール
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY app/ .

# デフォルトで uvicorn を実行
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
