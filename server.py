# -*- coding: utf-8 -*-
"""Qwen3-Embedding-0.6B CPU 推理服务：OpenAI 兼容 /v1/embeddings 接口"""
import os
import threading
from typing import List, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast

# 模型目录默认跟随脚本所在位置（部署目录变化不影响），可用环境变量覆盖
_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3-Embedding-0.6B")
MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", _DEFAULT_MODEL_DIR)
HOST = os.environ.get("EMBEDDING_HOST", "0.0.0.0")
PORT = int(os.environ.get("EMBEDDING_PORT", "8000"))
# 推理设备：cpu（默认，本地 WSL）| npu（昇腾生产，需环境已装 torch_npu）
DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
# NPU 上用 float16 权重（0.6B 显存 2.4GB→1.2GB，官方教程同款精度），CPU 保持 float32
DTYPE = getattr(torch, os.environ.get("EMBEDDING_DTYPE", "float16" if DEVICE.startswith("npu") else "float32"))
MAX_BATCH = 128
MIN_DIM, MAX_DIM = 32, 1024  # Qwen3-Embedding MRL 支持范围

if DEVICE.startswith("npu"):
    import torch_npu  # noqa: F401  import 后 torch 才注册 npu 后端

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "8")))
model = SentenceTransformer(MODEL_PATH, device=DEVICE, model_kwargs={"torch_dtype": DTYPE})
tokenizer: PreTrainedTokenizerFast = model.tokenizer
DIM = model.get_sentence_embedding_dimension()

app = FastAPI(title="Qwen3-Embedding CPU Service", version="1.0.0")
_encode_lock = threading.Lock()  # ST encode 非线程安全，串行化推理


class EmbeddingRequest(BaseModel):
    model: str = "qwen3-embedding-0.6b"
    input: Union[str, List[str]] = Field(..., description="文本或文本数组")
    dimensions: int = Field(default=DIM, ge=MIN_DIM, le=MAX_DIM, description="MRL 动态维度")
    encoding_format: str = Field(default="float", pattern="^(float|base64)$")


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: Union[List[float], str]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingItem]
    model: str
    usage: dict


@app.get("/v1/models")
def list_models():
    """OpenAI 标准模型列表接口，供平台「可选拉取」功能调用"""
    import time

    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-embedding-0.6b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "context_length": model.max_seq_length,  # 后端 RemoteModelListService 读取此字段
            }
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH), "dim": DIM, "device": DEVICE, "dtype": str(DTYPE)}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise HTTPException(status_code=400, detail="input 不能为空")
    if len(texts) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"单批最多 {MAX_BATCH} 条")

    with _encode_lock:
        vectors = model.encode(
            texts,
            normalize_embeddings=True,  # 归一化向量，余弦/内积检索直接可用
            truncate_dim=req.dimensions,
            show_progress_bar=False,
        )

    import base64
    import struct

    data = []
    total_tokens = 0
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        if req.encoding_format == "base64":
            emb = base64.b64encode(struct.pack(f"<{len(vec)}f", *vec.tolist())).decode()
        else:
            emb = vec.tolist()
        data.append(EmbeddingItem(index=i, embedding=emb))
        total_tokens += len(tokenizer(text)["input_ids"])

    return EmbeddingResponse(
        data=data,
        model=req.model,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )


if __name__ == "__main__":
    # 强制 h11 实现：httptools 解析器遇到 Upgrade 头（如 Java HttpClient 的 h2c 升级头）
    # 会抛 HttpParserUpgrade 并停止解析 body，uvicorn 对非 websocket 升级直接丢弃，
    # 导致服务端返回 422 空 body。h11 对该类请求可正常解析（实测，见部署文档）。
    uvicorn.run(app, host=HOST, port=PORT, http="h11", log_level="info")
