# embedding-service

OpenAI 兼容的 [Qwen3-Embedding-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B) 推理服务：**SentenceTransformer + torch_npu（昇腾 NPU）/ CPU**，FastAPI + uvicorn 实现，专为「大模型与 embedding 共卡部署」场景设计。

## 为什么不用 vLLM

vLLM 的 `gpu_memory_utilization`（默认 0.9）按**整卡**比例预算显存，不感知卡上已被其他进程（如 DeepSeek）占用的部分——共卡场景启动即失败，且 vllm-ascend 在单卡多服务下有已知问题链（[#7308](https://github.com/vllm-project/vllm-ascend/issues/7308)、[#5707](https://github.com/vllm-project/vllm-ascend/issues/5707)）。

而 embedding 是"一进一出"的单次前向推理：无 KV cache 累积、权重多大就占多大（fp16 仅 1.2GB）、平台侧低频串行调用。**不需要 vLLM 的高并发生成调度，直接用容器里现成的 torch_npu 跑 SentenceTransformer**，uvicorn 做 HTTP 并发即可。

## 特性

- **OpenAI 规范接口**：`/v1/embeddings`（float / base64、`dimensions` 动态维度、usage 统计）+ `/v1/models` + `/health`，平台侧零改造接入
- **CPU / NPU 双模式**：同一份代码，`EMBEDDING_DEVICE` 环境变量切换；NPU 上自动使用 float16（显存减半），CPU 上 float32
- **MRL 动态维度**：Qwen3-Embedding 支持 32~1024 维 Matryoshka 截断，按需返回
- **归一化输出**：所有向量 L2 归一，余弦/内积检索直接可用
- **串行推理锁**：SentenceTransformer 非线程安全，内置锁串行化 encode
- **离线部署**：5 个缺失依赖的 wheel 离线安装（`--no-index` 模式，绝不升级容器已有包，torch_npu 配对零扰动），一键脚本安装 + 启动 + 健康检查

## 快速开始

### 1. 下载模型

```bash
# ModelScope（国内推荐）
pip install modelscope
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir models/Qwen3-Embedding-0.6B

# 或 HuggingFace
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir models/Qwen3-Embedding-0.6B
```

### 2. 安装依赖

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # CPU 环境
```

> NPU 环境**不要**用 requirements.txt 覆盖安装（会破坏容器内 torch/torch_npu 配对）。
> 正确做法：沿用容器已有的 torch_npu，只补缺失的 5 个包，见下文「NPU 离线部署」。

### 3. 启动

```bash
# CPU（默认 8000 端口，float32）
./service.sh start

# NPU（昇腾容器内，8001 端口，float16，指定卡号）
EMBEDDING_DEVICE=npu EMBEDDING_PORT=8001 ASCEND_RT_VISIBLE_DEVICES=5 \
  nohup python server.py > server.log 2>&1 &

# 验证
curl http://127.0.0.1:8001/health
# {"status":"ok","model":"Qwen3-Embedding-0.6B","dim":1024,"device":"npu","dtype":"torch.float16"}
```

## API

### POST /v1/embeddings

```bash
curl http://127.0.0.1:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding-0.6b","input":["测试"],"dimensions":1024}'
```

响应遵循 OpenAI 规范（`data[].embedding` 向量数组 + `usage` token 统计）。`input` 支持字符串或数组（单批上限 128 条），`dimensions` 支持 32~1024。

### GET /v1/models

返回模型列表及 `context_length`，供平台的"模型列表拉取"功能调用。

### GET /health

存活探针，返回模型名、维度、设备与精度，适合做容器健康检查。

## 配置项（环境变量）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EMBEDDING_MODEL_PATH` | `<脚本目录>/models/Qwen3-Embedding-0.6B` | 模型目录（默认跟随脚本位置） |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` 或 `npu` |
| `EMBEDDING_DTYPE` | NPU: `float16` / CPU: `float32` | 权重精度 |
| `EMBEDDING_HOST` | `0.0.0.0` | 监听地址 |
| `EMBEDDING_PORT` | `8000` | 监听端口 |
| `TORCH_NUM_THREADS` | `8` | CPU 推理线程数 |
| `ASCEND_RT_VISIBLE_DEVICES` | 未设置 | NPU 卡号（设为剩余显存最多的卡） |

固定常量（`server.py`）：单批上限 128 条，MRL 维度范围 32~1024。

## NPU 离线部署（生产环境不通外网）

在 DeepSeek 等已有的昇腾容器内部署（复用其 CANN / torch_npu 环境），四步：

```bash
# ① 在有网机器上按目标容器规格下载缺失依赖（Python 版本/架构按实际改）
pip download sentence-transformers scikit-learn joblib narwhals threadpoolctl \
  -d wheels --no-deps --only-binary=:all: \
  --python-version 311 --platform manylinux2014_aarch64 --implementation cp

# ② 容器内离线安装：--no-index 保证只装缺失的 5 个包，已有包（torch/torch_npu/transformers）零改动
pip install --no-index --find-links=wheels sentence-transformers scikit-learn

# ③ 验证 torch_npu 配对未被破坏
python -c "import sentence_transformers, torch, torch_npu; print(torch.npu.is_available())"

# ④ 启动（或使用一键脚本：CARD_ID=<卡号> ./install-and-start.sh）
```

一键脚本 `install-and-start.sh` 支持 `start`（默认）/ `stop` / `status`，内置健康轮询与幂等处理（容器重建后重跑即可）。

## 已知坑：Java HttpClient 调用返回 422 空 body

**现象**：Java `HttpClient`（默认 HTTP/2 意向）调用本服务时，FastAPI 报 `422 {"loc":["body"],"msg":"Field required"}`。

**根因**：Java 端对明文 HTTP 地址发送 h2c 升级头（`Connection: Upgrade` + `Upgrade: h2c`）；uvicorn 装有 httptools 时，其 C 解析器一见到 Upgrade 头就停止解析 body（按协议升级语义），uvicorn 对非 websocket 升级打一行 `Unsupported upgrade request` 后直接丢弃请求体。

**修复**（本仓库已内置）：`uvicorn.run(..., http="h11")` 强制使用 h11 HTTP 实现，对升级头请求可正常解析。服务端日志出现 `Unsupported upgrade request` 即可确认命中此坑。

## 目录结构

```
embedding-service/
├── server.py               # 服务主体：OpenAI 兼容接口 + ST 推理（约 120 行）
├── validate_model.py       # 模型效果验证：1024 维断言 / 语义相似度 / MRL 截断 / 归一化
├── install-and-start.sh    # NPU 容器一键离线安装 + 启动（幂等）
├── service.sh              # 本地 CPU 环境管理脚本（start/stop/status）
├── service-npu.sh          # 独立 NPU 容器方案（build/run/stop/status，备选）
├── Dockerfile.npu          # 在线构建镜像（基于 vllm-ascend 官方镜像）
├── Dockerfile.npu.offline  # 离线构建镜像（wheels/ 就地构建）
├── requirements.txt        # CPU 环境依赖
├── models/                 # 模型权重（不入库，见上方下载指引）
└── wheels/                 # 离线 wheel（默认不入库，见 .gitignore 说明）
```

## License

仅供内部学习交流使用。模型权重遵循 [Qwen3 官方许可](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B)。
