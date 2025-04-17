from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import uvicorn
import logging

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPIアプリケーションのインスタンスを作成
app = FastAPI(title="Japanese Embeddings and Reranker API")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GPUが利用可能かどうかを確認
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if device == "cuda":
    logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# モデルをロード
# サポートする埋め込みモデルのロード
default_embedding_model_name = "cl-nagoya/ruri-large-v2"
embedding_models = {
    default_embedding_model_name: SentenceTransformer(default_embedding_model_name),
    "cl-nagoya/ruri-v3-pt-310m": SentenceTransformer("cl-nagoya/ruri-v3-pt-310m"),
}
for model in embedding_models.values():
    model.to(device)

# サポートするリランカーモデルのロード
default_reranker_model_name = "cl-nagoya/ruri-reranker-large"
reranker_models = {
    default_reranker_model_name: CrossEncoder(default_reranker_model_name),
    "cl-nagoya/ruri-v3-reranker-310m": CrossEncoder("cl-nagoya/ruri-v3-reranker-310m"),
}
for model in reranker_models.values():
    if device == "cuda":
        model.model.to(device)

# OpenAI互換のリクエスト形式
class RerankRequest(BaseModel):
    model: str
    input: Optional[str] = None
    query: Optional[str] = None
    documents: List[str]
    max_chunks_per_doc: Optional[int] = None
    return_documents: Optional[bool] = True
    top_n: Optional[int] = None

    # 互換性のためにqueryとinputを処理
    def get_query(self) -> str:
        if self.query is not None:
            return self.query
        elif self.input is not None:
            return self.input
        else:
            raise ValueError("Either 'query' or 'input' must be provided")

@app.get("/")
def read_root():
    return {
        "message": "Japanese Embeddings and Reranker API is running",
        "embedding_model": "cl-nagoya/ruri-large-v2",
        "reranker_model": "cl-nagoya/ruri-reranker-large",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

# デバッグ用エンドポイント
@app.post("/debug")
async def debug_request(request: Request):
    """デバッグ用: リクエストの内容をそのまま返す"""
    body = await request.json()
    return {
        "received_request": body,
        "content_type": request.headers.get("content-type")
    }

# 埋め込み用エンドポイント - OpenAI互換
@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    try:
        body = await request.json()
        model_name = body.get("model", default_embedding_model_name)
        if model_name not in embedding_models:
            raise HTTPException(status_code=400, detail=f"Unsupported embedding model: {model_name}")
        model = embedding_models[model_name]
        input_texts = body.get("input", [])
        
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        
        # 必要に応じてクエリプレフィックスを追加
        processed_texts = []
        for text in input_texts:
            if not text.startswith("クエリ: ") and not text.startswith("文章: "):
                # デフォルトでは文章側として扱う
                text = "文章: " + text
            processed_texts.append(text)
        
        # embeddings生成
        embeddings = model.encode(processed_texts)
        embeddings_list = embeddings.tolist()
        
        # OpenAI互換のレスポンス形式
        response = {
            "data": [
                {
                    "embedding": emb,
                    "index": i,
                    "object": "embedding"
                } for i, emb in enumerate(embeddings_list)
            ],
            "model": model_name,
            "object": "list",
            "usage": {
                "prompt_tokens": len(" ".join(processed_texts).split()),
                "total_tokens": len(" ".join(processed_texts).split())
            }
        }
        
        return response
    except Exception as e:
        logger.error(f"Embeddings error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# 再ランキング用のエンドポイント - OpenAI互換
@app.post("/v1/rerank")
async def rerank(request: RerankRequest):
    try:
        # リクエスト内容のログ記録
        logger.info(f"Received rerank request: {request}")
        
        # モデルの選択
        model_name = request.model or default_reranker_model_name
        if model_name not in reranker_models:
            raise HTTPException(status_code=400, detail=f"Unsupported reranker model: {model_name}")
        model = reranker_models[model_name]
        
        # クエリと文書のペアを作成
        query = request.get_query()
        pairs = [[query, doc] for doc in request.documents]
                
        # スコアリング
        scores = model.predict(pairs)
        
        # 結果をフォーマット - OpenAI互換形式に変更
        results = []
        for i, score in enumerate(scores):
            results.append({
                "index": i,
                "relevance_score": float(score),
                "document": request.documents[i] if request.return_documents else None,
            })
        
        # スコアで降順ソート
        results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        
        # top_nが指定されている場合は上位N件のみ返す
        if request.top_n:
            results = results[:request.top_n]
            
        # OpenAI互換のレスポンス形式
        response = {
            "object": "list",
            "model": model_name,
            "results": results,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        }
        
        logger.info(f"Returning response: {response}")
        return response
    except Exception as e:
        logger.error(f"Rerank error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rerank")
async def rerank(request: RerankRequest):
    try:
        # リクエスト内容のログ記録
        logger.info(f"Received rerank request: {request}")
        
        # モデルの選択
        model_name = request.model or default_reranker_model_name
        if model_name not in reranker_models:
            raise HTTPException(status_code=400, detail=f"Unsupported reranker model: {model_name}")
        model = reranker_models[model_name]
        
        # クエリと文書のペアを作成
        query = request.get_query()
        pairs = [[query, doc] for doc in request.documents]
                
        # スコアリング
        scores = model.predict(pairs)
        
        # 結果をフォーマット - OpenAI互換形式に変更
        results = []
        for i, score in enumerate(scores):
            results.append({
                "index": i,
                "relevance_score": float(score),
                "document": request.documents[i] if request.return_documents else None,
            })
        
        # スコアで降順ソート
        results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        
        # top_nが指定されている場合は上位N件のみ返す
        if request.top_n:
            results = results[:request.top_n]
            
        # OpenAI互換のレスポンス形式
        response = {
            "object": "list",
            "model": model_name,
            "results": results,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        }
        
        logger.info(f"Returning response: {response}")
        return response
    except Exception as e:
        logger.error(f"Rerank error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# 類似度計算エンドポイント - 独自API
@app.post("/similarity")
async def calculate_similarity(request: Request):
    data = await request.json()
    texts = data.get("texts", [])

    if len(texts) < 2:
        return {"error": "At least 2 texts are required to calculate similarity"}

    # 必要に応じてプレフィックスを追加
    processed_texts = []
    for text in texts:
        if not text.startswith("クエリ: ") and not text.startswith("文章: "):
            # デフォルトでは文章側として扱う
            text = "文章: " + text
        processed_texts.append(text)

    # embeddings生成
    embeddings = embedding_models[default_embedding_model_name].encode(processed_texts, convert_to_tensor=True)

    # コサイン類似度の計算
    similarity_matrix = F.cosine_similarity(
        embeddings.unsqueeze(0),
        embeddings.unsqueeze(1),
        dim=2
    ).cpu().numpy().tolist()

    return {
        "similarity_matrix": similarity_matrix,
        "texts": processed_texts,
        "device": device
    }

# pipeline_tag: sentence-similarity エンドポイント
@app.post("/sentence-similarity")
async def sentence_similarity_endpoint(request: Request):
    # 既存の /similarity ロジックを再利用
    return await calculate_similarity(request)

# pipeline_tag: text-ranking エンドポイント
@app.post("/text-ranking")
async def text_ranking_endpoint(request: Request):
    data = await request.json()
    query = data.get("query")
    corpus = data.get("corpus")
    if query is None or corpus is None:
        raise HTTPException(status_code=400, detail="Both 'query' and 'corpus' must be provided")
    if isinstance(corpus, str):
        corpus = [corpus]
    # v3モデルを使用してリランキング
    model = reranker_models["cl-nagoya/ruri-v3-reranker-310m"]
    results = model.rank(query=query, documents=corpus)
    return {
        "results": results
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "embedding_model": "cl-nagoya/ruri-large-v2",
        "reranker_model": "cl-nagoya/ruri-reranker-large"
    }

# サーバー起動用のコード
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
