# qwen3-embedding-ascend

**昇腾 NPU 空闲显存离线 Embedding 服务**：复用 [vllm-ascend](https://quay.io/repository/ascend/vllm-ascend) 官方容器镜像的 CANN/torch_npu 环境，借用 NPU 空闲显存运行 [Qwen3-Embedding-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B)，提供开箱即用的 OpenAI 兼容 `/v1/embeddings` 接口。**不依赖 vLLM 推理引擎**，专为「大模型主服务与 embedding 共卡部署、生产环境不通外网」场景设计。

## 解决什么问题

昇腾服务器上已经跑着 LLM 主服务（如 DeepSeek），显存占了大头但没占完；生产环境不通外网。需求：**用剩余显存跑一个知识库/RAG 向量化服务，不动主服务、不加卡、全程离线**。

方案：借 vllm-ascend 官方镜像当"底座"——CANN、torch、torch_npu、transformers 的版本矩阵已由官方配对好，拿来即用；但**不跑 vLLM 引擎**，容器内用 SentenceTransformer 直接推理：

- vLLM 的 `gpu_memory_utilization`（默认 0.9）按**整卡**比例预算显存，不感知卡上已被主服务占用的部分——共卡场景启动即失败（见 vllm-ascend [#7308](https://github.com/vllm-project/vllm-ascend/issues/7308)、[#5707](https://github.com/vllm-project/vllm-ascend/issues/5707)）；
- embedding 是"一进一出"的单次前向：无 KV cache 累积、fp16 权重仅 1.2GB（fp32 的一半）、低频串行调用——**权重多大就占多大，天然适合借用空闲显存**，用不到 vLLM 的高并发生成调度；
- HTTP 层用 FastAPI + uvicorn 提供 OpenAI 规范接口，平台侧零改造接入。

## 特性

- **OpenAI 规范接口**：`/v1/embeddings`（float / base64、`dimensions` 动态维度、usage 统计）+ `/v1/models` + `/health`，平台侧零改造接入
- **CPU / NPU 双模式**：同一份代码，`EMBEDDING_DEVICE` 环境变量切换；NPU 自动 float16（显存减半），CPU float32
- **MRL 动态维度**：Qwen3-Embedding 支持 32~1024 维 Matryoshka 截断，按需返回
- **归一化输出**：所有向量 L2 归一，余弦/内积检索直接可用
- **离线部署**：只补 5 个缺失依赖 wheel（`--no-index` 模式绝不升级镜像已有包，torch_npu 配对零扰动），一键脚本安装 + 启动 + 健康检查

## 基础镜像

| 项 | 值 |
|---|---|
| 镜像 | `quay.io/ascend/vllm-ascend:v0.20.2rc1` |
| 拉取 | `docker pull quay.io/ascend/vllm-ascend:v0.20.2rc1` |
| 镜像页 | <https://quay.io/repository/ascend/vllm-ascend?tab=tags> |
| 项目文档 | <https://docs.vllm.ai/projects/vllm-ascend/en/latest/> |

镜像内实测环境（本仓库 `wheels/` 依赖按此核对）：

```
Python 3.11.15（aarch64）  torch 2.10.0+cpu  torch_npu 2.10.0（严格配对）
vllm 0.20.2+empty  vllm_ascend 0.20.2rc1  transformers 5.5.3
fastapi 0.123.10  uvicorn 0.46.0  numpy 1.26.4  scipy 1.13.1
```

## 获取服务（两种方式）

| | 方式 A：Release 完整包（推荐） | 方式 B：clone 源码 |
|---|---|---|
| 内容 | **模型（1.2GB）+ wheels + 全部脚本**，开箱即用 | 代码 + wheels（**不含模型**） |
| 适用 | NPU 生产离线部署、快速试用 | 二次开发、CPU 本地调试 |
| 获取 | [GitHub Releases](../../releases) 下载 `embedding-service-npu.tar.gz` | `git clone` 本仓库 |

> Release 包不进 git 历史，仓库克隆保持轻量；单文件 916MB 在 GitHub Release 2GB 限额内。

### 模型获取说明（CPU 和 NPU 都需要模型，差异只在方式）

| 场景 | 获取方式 |
|---|---|
| CPU 本地（通网） | 在线下载（二选一）：`modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir models/Qwen3-Embedding-0.6B`，或 `huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local_dir models/Qwen3-Embedding-0.6B` |
| NPU 生产（离线） | ① **Release 完整包已含模型**，解压即用（推荐）；② 或在有网机器下载后 `scp` 到生产机数据目录 |

## 目录结构

```
├── server.py               # 服务主体：FastAPI + uvicorn(http="h11")，CPU/NPU 双模式（约 120 行）
├── docker-create.sh        # NPU 宿主机：启动 vllm-ascend 镜像容器（设备/驱动挂载）
├── install-and-start.sh    # NPU 容器内：离线装依赖 → 校验 torch_npu → 启动 → 健康检查（幂等）
├── validate_model.py       # 模型验证：1024 维 / 语义相似度 / MRL 截断 / 归一化 四断言
├── service.sh              # 本地 CPU：start/stop/status 管理脚本
├── requirements.txt        # 本地 CPU 依赖（NPU 用容器环境 + wheels/，勿在 NPU 容器安装）
├── wheels/                 # NPU 离线依赖包（cp311/aarch64，5 个）
├── Dockerfile.npu(.offline)  # 备选：自制镜像方案（在线/离线构建）
├── service-npu.sh          # 备选：自制镜像方案的管理脚本
└── models/                 # 模型权重（1.2GB，不入库；获取方式见「模型获取说明」）
```

## 本地 CPU 快速开始

```bash
# 1. 获取模型（见上方「模型获取说明」，通网机器在线下载）
pip install modelscope
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir models/Qwen3-Embedding-0.6B

# 2. 装依赖（CPU 版 torch 即可；NPU 容器用 wheels/ 离线安装，勿执行本步）
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 3. 验证模型（四断言：1024 维 / 语义相似度 / MRL 截断 / 归一化）
python validate_model.py

# 4. 启动（默认 cpu + 8000 端口）
./service.sh start && ./service.sh status
```

## NPU 部署（三步）

前置：宿主机已装昇腾驱动（`npu-smi info` 可用）、已拉取基础镜像、已通过上方「获取服务」拿到**完整包**（Release 方式 A）或将源码与模型放到数据目录。

### ① 启动容器（生产实测命令）

```bash
docker run -itd --privileged --name=qwen3-embeding-vllm20 -p 8090:8001 \
  --shm-size 500g \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  --device=/dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /data/qwen3-embeding/:/root/ \
  -w /root/ \
  quay.io/ascend/vllm-ascend:v0.20.2rc1 \
  bash
```

> 仓库内同款脚本 [`docker-create.sh`](docker-create.sh)（`HOST_PORT`/`DATA_DIR` 可用环境变量覆盖）。`davinci0-7` 按宿主机实际卡数与空闲情况裁剪；`--shm-size` 沿用生产实配可调小；数据目录挂为容器 `/root/`。

### ② 容器内离线安装并启动

```bash
docker exec -it qwen3-embeding-vllm20 bash
cd /root/embedding-service

# 卡号选 npu-smi info 里剩余显存最多的卡
CARD_ID=0 ./install-and-start.sh
```

> Release 完整包解压后即为此目录结构（`/data/qwen3-embeding/embedding-service/`，挂载为容器 `/root/embedding-service`）。若用方式 B clone，则把仓库内容与模型目录放到数据目录下。

脚本自动完成：离线安装缺失依赖（`pip install --no-index --find-links=wheels`，已有包一律不动）→ 校验 torch_npu 配对（打印 `npu可用: True`）→ 以 `EMBEDDING_DEVICE=npu` + float16 启动 → 60 秒健康轮询。管理：`./install-and-start.sh stop|status`，日志 `tail -f server.log`。容器重建后重跑即可（幂等）。

### ③ 验证

```bash
curl -s http://127.0.0.1:8001/health
# {"status":"ok","model":"Qwen3-Embedding-0.6B","dim":1024,"device":"npu","dtype":"torch.float16"}

curl -s http://127.0.0.1:8001/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding-0.6b","input":["测试"]}'

npu-smi info   # 目标卡显存上涨约 1.2GB
```

调用方接入地址：`http://<宿主机IP>:8090/v1/embeddings`（容器 8001 已映射宿主机 8090）。

## API

### POST /v1/embeddings

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding-0.6b","input":["测试"],"dimensions":1024}'
```

响应遵循 OpenAI 规范（`data[].embedding` + `usage` token 统计）。`input` 支持字符串或数组（单批上限 128 条），`dimensions` 支持 32~1024，`encoding_format` 支持 `float`/`base64`。

### GET /v1/models

模型列表及 `context_length`，供平台的"模型列表拉取"功能调用。

### GET /health

存活探针，返回模型名、维度、设备与精度。

## 配置项（环境变量）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EMBEDDING_MODEL_PATH` | `<脚本目录>/models/Qwen3-Embedding-0.6B` | 模型目录（跟随脚本位置） |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` 或 `npu` |
| `EMBEDDING_DTYPE` | NPU: `float16` / CPU: `float32` | 权重精度 |
| `EMBEDDING_HOST` | `0.0.0.0` | 监听地址 |
| `EMBEDDING_PORT` | `8000` | 监听端口（NPU 脚本默认 8001） |
| `TORCH_NUM_THREADS` | `8` | CPU 推理线程数 |
| `ASCEND_RT_VISIBLE_DEVICES` | 未设置 | NPU 卡号（`install-and-start.sh` 的 `CARD_ID` 会注入） |

固定常量（`server.py`）：单批上限 128 条，MRL 维度范围 32~1024。

## FAQ

**Q: Java 服务调用报 422 `{"loc":["body"],"msg":"Field required"}`？**
A: Java HttpClient（默认 HTTP/2 意向）对明文 HTTP 发 h2c 升级头（`Connection: Upgrade` + `Upgrade: h2c`）；uvicorn 装有 httptools 时，其 C 解析器按协议升级语义停止解析 body，uvicorn 对非 websocket 升级打一行 `Unsupported upgrade request` 后丢弃请求体。**本仓库 `server.py` 已内置修复**：`uvicorn.run(..., http="h11")` 强制 h11 实现（实测 422→200）。Java 侧可另加 `HttpClient.Version.HTTP_1_1` 双保险；服务端日志出现 `Unsupported upgrade request` 即命中此坑。

**Q: 调用方报「速率限制排队超时」？**
A: 这是 aigc-app 知识库管线自己的限流隔离舱（默认 2 次/秒），非本服务问题；多文档并行向量化时可调大调用方 `knowledge.pipeline.bulkhead.embedding.*` 参数。

**Q: NPU 显存不足？**
A: fp16 权重约 1.2GB，建议目标卡空闲 ≥2.5GB；`npu-smi info` 选剩余最多的卡，`CARD_ID=<卡号>` 指定。

**Q: 容器重建后服务没了？**
A: pip 安装在容器可写层，重建后重跑 `CARD_ID=x ./install-and-start.sh` 即可（幂等）。

## License

MIT。模型权重遵循 [Qwen3 官方许可](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B)。

MIT。模型权重遵循 [Qwen3 官方许可](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B)。
