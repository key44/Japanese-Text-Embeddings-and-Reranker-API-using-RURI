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

# GPUが利用可能か確認するスクリプトを追加
RUN echo '#!/bin/bash\n\
echo "Checking CUDA availability..."\n\
python -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")"\n\
if [ $? -eq 0 ]; then\n\
  echo "CUDA is available. Starting the API server..."\n\
  exec uvicorn main:app --host 0.0.0.0 --port 8000\n\
else\n\
  echo "WARNING: CUDA is not available. The API will run on CPU mode."\n\
  exec uvicorn main:app --host 0.0.0.0 --port 8000\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# スタートスクリプトを実行
CMD ["/app/start.sh"]
