For a Japanese version of this README, see [README_ja.md](README_ja.md).

# Japanese Embeddings and Reranker API

This is a FastAPI application providing Japanese text embeddings and reranking capabilities. It is based on **Ruri: Japanese General Text Embeddings** developed by the Sasano Laboratory at Nagoya University's Graduate School of Informatics, Value Creation Research Center.

## Features

*   **Embeddings:** Generate embeddings for Japanese text using the `cl-nagoya/ruri-large-v2` model. The API provides an OpenAI-compatible `/v1/embeddings` endpoint.
*   **Reranking:** Rerank search results or other document lists based on a query using the `cl-nagoya/ruri-reranker-large` model. The API provides an OpenAI-compatible `/v1/rerank` endpoint.
*   **Similarity Calculation:** Calculate the cosine similarity between texts using a custom `/similarity` endpoint.
*   **Health Check:** A `/health` endpoint provides information about the API's status, including the models used and device availability (CPU/CUDA).
*   **Dockerized:** The application is designed to be run within a Docker container, making deployment and scaling easy.
*   **GPU Support:** The application automatically utilizes a GPU if available, falling back to CPU if not.

## Models

This application utilizes the following models:

*   **Embedding Model:** `cl-nagoya/ruri-large-v2` ([Hugging Face](https://huggingface.co/cl-nagoya/ruri-large-v2))
*   **Reranker Model:** `cl-nagoya/ruri-reranker-large` ([Hugging Face](https://huggingface.co/cl-nagoya/ruri-reranker-large))

These models are based on the research presented in:

*   **Ruri: Japanese General Text Embeddings** ([arXiv](https://arxiv.org/abs/2409.07737))

## Prerequisites

*   Docker
*   (Optional) NVIDIA GPU and NVIDIA Container Toolkit for GPU acceleration.

## Usage

### Building and Running with Docker

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

    (Replace `<repository_url>` and `<repository_directory>` with the actual values.)

2.  **Build the Docker image:**

    ```bash
    docker compose build
    ```

3.  **Run the application:**

    ```bash
    docker compose up -d
    ```

    This will start the API server in detached mode. The API will be accessible on port 8100 (as defined in `docker-compose.yml`).

4.  **Stopping the application**

    ```bash
    docker compose down
    ```

### API Endpoints

The API provides the following endpoints:

*   **`/` (GET):** Returns basic information about the running API, including the models used and device information.

    Example Response:

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

*   **`/v1/embeddings` (POST):** Generates embeddings for a given input text. This endpoint is designed to be compatible with the OpenAI embeddings API.

    Request Body (example):

    ```json
    {
        "input": ["文章: これはテストです", "クエリ: 日本語の埋め込み"],
        "model": "cl-nagoya/ruri-large-v2"
    }
    ```

    *   You can provide a string or a list of strings.
    *   The API automatically prepends "文章: " or "クエリ: " if not provided.

    Response Body (example):

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

*   **`/v1/rerank` (POST):** Reranks a list of documents based on a query. This endpoint is designed to be compatible with the OpenAI reranking API.

    Request Body (example):

    ```json
    {
      "query": "類似度計算",
      "documents": ["ドキュメント1", "類似度を計算する方法", "FastAPIについて"],
      "model": "cl-nagoya/ruri-reranker-large",
      "top_n": 2
    }
    ```

    *   `query` or `input` is required.
    *   `documents`: List of documents to rerank.
    *   `top_n`: (Optional) Return only the top N results.
    *   `return_documents`: (Optional, default: true) Whether to return the documents in the response.

    Response Body (example):
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

*   **`/similarity` (POST):** Calculates the cosine similarity between a list of texts. This is a custom endpoint.
*   **`/sentence-similarity` (POST):** Reuses `/similarity` logic under the `sentence-similarity` pipeline tag.
*   **`/text-ranking` (POST):** Accepts `query` and `corpus`, ranks documents using the `text-ranking` pipeline tag with the `cl-nagoya/ruri-v3-reranker-310m` model.

    Request Body (example):

    ```json
    {
        "texts": ["文章: テキスト1", "文章: テキスト2"]
    }
    ```

    *   The API automatically prepends "文章: " if not provided.

    Response Body (example):

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

*   **`/health` (GET):** Returns the health status of the API.

    Example Response:

    ```json
    {
        "status": "healthy",
        "device": "cuda",
        "cuda_available": true,
        "embedding_model": "cl-nagoya/ruri-large-v2",
        "reranker_model": "cl-nagoya/ruri-reranker-large"
    }
    ```

*   **`/debug` (POST):** An endpoint for debugging. It returns the received request.

### Docker Compose Configuration

The `docker-compose.yml` file configures the application to use a Docker-managed volume (`model-cache`) to store the downloaded Sentence Transformers models. This prevents re-downloading the models every time the container is rebuilt. It also configures GPU usage if available. The API is exposed on port 8100 on the host machine.

## Acknowledgments

We would like to thank the Sasano Laboratory at Nagoya University for their work on Ruri: Japanese General Text Embeddings, which this project is based on.

*   **Sasano Laboratory:** [http://cr.fvcrc.i.nagoya-u.ac.jp/](http://cr.fvcrc.i.nagoya-u.ac.jp/)