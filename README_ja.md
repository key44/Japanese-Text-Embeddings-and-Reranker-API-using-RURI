# 日本語埋め込みとリランキングAPI

これは、FastAPIを使用して日本語テキストの埋め込みとリランキング機能を提供するアプリケーションです。名古屋大学大学院情報学研究科価値創造研究センター笹野研究室による**Ruri: Japanese General Text Embeddings**に基づいています。

## 機能

*   **埋め込み:** `cl-nagoya/ruri-large-v2`モデルを使用して日本語テキストの埋め込みを生成します。APIはOpenAI互換の`/v1/embeddings`エンドポイントを提供します。
*   **リランキング:** `cl-nagoya/ruri-reranker-large`モデルを使用して、クエリに基づいて検索結果やその他のドキュメントリストをリランキングします。APIはOpenAI互換の`/v1/rerank`エンドポイントを提供します。
*   **類似度計算:** カスタムの`/similarity`エンドポイントを使用して、テキスト間のコサイン類似度を計算します。
*   **ヘルスチェック:** `/health`エンドポイントは、使用されているモデルやデバイスの可用性（CPU/CUDA）など、APIのステータスに関する情報を提供します。
*   **Docker化:** アプリケーションはDockerコンテナ内で実行するように設計されており、デプロイとスケーリングが容易です。
*   **GPUサポート:** アプリケーションは、利用可能な場合は自動的にGPUを利用し、利用できない場合はCPUにフォールバックします。

## モデル

このアプリケーションは、次のモデルを利用しています。

*   **埋め込みモデル:** `cl-nagoya/ruri-large-v2` ([Hugging Face](https://huggingface.co/cl-nagoya/ruri-large-v2))
*   **リランカーモデル:** `cl-nagoya/ruri-reranker-large` ([Hugging Face](https://huggingface.co/cl-nagoya/ruri-reranker-large))

これらのモデルは、以下の研究に基づいています。

*   **Ruri: Japanese General Text Embeddings** ([arXiv](https://arxiv.org/abs/2409.07737))

## 前提条件

*   Docker
*   (オプション) NVIDIA GPUおよびGPUアクセラレーション用のNVIDIA Container Toolkit。

## 使用方法

### Dockerを使用したビルドと実行

1.  **リポジトリをクローンします:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

    (`<repository_url>`と`<repository_directory>`を実際の値に置き換えてください。)

2.  **Dockerイメージをビルドします:**

    ```bash
    docker compose build
    ```

3.  **アプリケーションを実行します:**

    ```bash
    docker compose up -d
    ```

    これにより、APIサーバーがデタッチモードで起動します。APIはポート8100でアクセスできます（`docker-compose.yml`で定義）。

4.  **アプリケーションの停止**

    ```bash
    docker compose down
    ```

### APIエンドポイント

APIは次のエンドポイントを提供します。

*   **`/` (GET):** 実行中のAPIに関する基本情報（使用されているモデルやデバイス情報など）を返します。

    応答例：

    ```json
    {
        "message": "Japanese Embeddings and Reranker API is running",
        "embedding_model": "cl-nagoya/ruri-large-v2",
        "reranker_model": "cl-nagoya/ruri-reranker-large",
        "device": "cuda",
        "cuda_available": true,
        "gpu_info": "NVIDIA GeForce RTX 3090"
    }
    ```

*   **`/v1/embeddings` (POST):** 指定された入力テキストの埋め込みを生成します。このエンドポイントは、OpenAIの埋め込みAPIと互換性があるように設計されています。

    リクエストボディ（例）：

    ```json
    {
        "input": ["文章: これはテストです", "クエリ: 日本語の埋め込み"],
        "model": "cl-nagoya/ruri-large-v2"
    }
    ```

    *   文字列または文字列のリストを指定できます。
    *   APIは、指定されていない場合、自動的に"文章: "または"クエリ: "を前置します。

    応答ボディ（例）：

    ```json
    {
        "data": [
            {
                "embedding": [0.1, 0.2, ...],
                "index": 0,
                "object": "embedding"
            },
            {
                "embedding": [0.3, 0.4, ...],
                "index": 1,
                "object": "embedding"
            }
        ],
        "model": "cl-nagoya/ruri-large-v2",
        "object": "list",
        "usage": {
            "prompt_tokens": 10,
            "total_tokens": 10
        }
    }
    ```

*   **`/v1/rerank` (POST):** クエリに基づいてドキュメントのリストをリランキングします。このエンドポイントは、OpenAIのリランキングAPIと互換性があるように設計されています。

    リクエストボディ（例）：

    ```json
    {
      "query": "類似度計算",
      "documents": ["ドキュメント1", "類似度を計算する方法", "FastAPIについて"],
      "model": "cl-nagoya/ruri-reranker-large",
      "top_n": 2
    }
    ```

    *   `query`または`input`が必要です。
    *   `documents`: リランキングするドキュメントのリスト。
    *   `top_n`: (オプション) 上位N件のみを返します。
    *   `return_documents`: (オプション、デフォルト: true) レスポンスにドキュメントを含めるかどうか。

    応答ボディ（例）：

    ```json
    {
        "object": "list",
        "model": "cl-nagoya/ruri-reranker-large",
        "results": [
            {
                "index": 1,
                "relevance_score": 0.9,
                "document": "類似度を計算する方法"
            },
            {
                "index": 2,
                "relevance_score": 0.2,
                "document": "FastAPIについて"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    }
    ```

*   **`/similarity` (POST):** テキストのリスト間のコサイン類似度を計算します。これはカスタムエンドポイントです。

    リクエストボディ（例）：

    ```json
    {
        "texts": ["文章: テキスト1", "文章: テキスト2"]
    }
    ```

    *   APIは、指定されていない場合、自動的に"文章: "を前置します。

    応答ボディ（例）：

    ```json
    {
        "similarity_matrix": [
            [1.0, 0.8],
            [0.8, 1.0]
        ],
        "texts": ["文章: テキスト1", "文章: テキスト2"],
        "device": "cuda"
    }
    ```

*   **`/health` (GET):** APIのヘルスステータスを返します。

    応答例：

    ```json
    {
        "status": "healthy",
        "device": "cuda",
        "cuda_available": true,
        "embedding_model": "cl-nagoya/ruri-large-v2",
        "reranker_model": "cl-nagoya/ruri-reranker-large"
    }
    ```

*   **`/debug` (POST):** デバッグ用のエンドポイント。受信したリクエストを返します。

### Docker Composeの設定

`docker-compose.yml`ファイルは、ダウンロードしたSentence Transformersモデルを保存するためにDocker管理ボリューム（`model-cache`）を使用するようにアプリケーションを構成します。これにより、コンテナが再構築されるたびにモデルを再ダウンロードする必要がなくなります。また、利用可能な場合はGPUの使用を構成します。APIはホストマシンのポート8100で公開されます。

## 謝辞

このプロジェクトの基盤となっているRuri: Japanese General Text Embeddingsの研究にご尽力いただいた名古屋大学笹野研究室に感謝いたします。

*   **笹野研究室:** [http://cr.fvcrc.i.nagoya-u.ac.jp/](http://cr.fvcrc.i.nagoya-u.ac.jp/)

このREADMEの英語版は[README.md](README.md)をご覧ください。