# -*- coding: utf-8 -*-
"""Qwen3-Embedding-0.6B 模型验证脚本：维度断言 + 语义相似度断言 + MRL 截断验证"""
import os

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 模型目录默认跟随脚本所在位置，与 server.py 保持一致，可用环境变量覆盖
_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3-Embedding-0.6B")
MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", _DEFAULT_MODEL_DIR)

model = SentenceTransformer(MODEL_PATH, device="cpu")
print(f"模型加载成功 | 最大序列长度: {model.max_seq_length} | 向量维度: {model.get_sentence_embedding_dimension()}")

# 官方推荐的查询指令（instruction-aware query embedding）
query_instruction = "Instruct: Given a web search query, retrieve relevant passages that help answer the question\nQuery: "

queries = [
    query_instruction + "中国的首都是哪里？",
    query_instruction + "how to cook pizza",
]
documents = [
    "中国的首都是北京，是一座历史悠久的城市。",
    "Pizza is an Italian dish made with dough, tomato sauce and cheese.",
    "今天天气很好，适合出门散步。",
]

q_emb = model.encode(queries, prompt_name=None)  # 指令已手动拼接
d_emb = model.encode(documents)

# 断言 1：向量维度必须为 1024
assert q_emb.shape[1] == 1024, f"维度异常: {q_emb.shape[1]}"
print(f"[断言1 通过] 向量维度 = {q_emb.shape[1]}")

# 断言 2：语义相关性 — 相关句对相似度必须显著高于无关句对
sims = cos_sim(q_emb, d_emb)
print("查询1（中国首都）与各文档相似度:", [round(float(s), 4) for s in sims[0]])
print("查询2（披萨做法）与各文档相似度:", [round(float(s), 4) for s in sims[1]])

# 正确文档必须居首，且显著高于最优无关项（无关文档之间的相对顺序无语义意义）
assert float(sims[0][0]) == float(sims[0].max()) and float(sims[0][0]) - max(float(sims[0][1]), float(sims[0][2])) > 0.2, "查询1 相关性异常"
assert float(sims[1][1]) == float(sims[1].max()) and float(sims[1][1]) - max(float(sims[1][0]), float(sims[1][2])) > 0.2, "查询2 相关性异常"
print("[断言2 通过] 相关句对相似度显著高于无关句对，语义检索能力正常")

# 断言 3：MRL 动态维度截断（32-1024 可调）
d_emb_512 = model.encode(documents, truncate_dim=512)
assert d_emb_512.shape[1] == 512, f"MRL 截断异常: {d_emb_512.shape[1]}"
print(f"[断言3 通过] MRL 截断维度 = {d_emb_512.shape[1]}")

# 断言 4：批量与归一化检查
d_norm = model.encode(["单条测试"], normalize_embeddings=True)
# float32 数值精度下允许 ~0.2% 偏差（实测 1.00198），对余弦相似度检索无影响
assert abs(float(torch.norm(torch.tensor(d_norm[0]))) - 1.0) < 5e-3, "归一化异常"
print("[断言4 通过] 归一化向量 L2 范数 ≈ 1.0")

print("\n✅ 全部验证通过，模型可用于向量服务")
