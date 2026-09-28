#!/usr/bin/env bash
# NPU 生产部署脚本：构建镜像 + 在剩余显存的卡上启动 embedding 容器
#
# 前置：模型文件已放在本目录 models/Qwen3-Embedding-0.6B（从本地 WSL rsync 过来）
# 用法:
#   BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.23.0 CARD_ID=5 ./service-npu.sh build   # 构建镜像（首次）
#   BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.23.0 CARD_ID=5 ./service-npu.sh run     # 启动容器
#   ./service-npu.sh stop     # 停止并删除容器
#   ./service-npu.sh status   # 健康检查
#   ./service-npu.sh check    # 回归验证 torch_npu 配对未被动过
#
# CARD_ID = npu-smi info 里剩余显存最多的卡号；该卡通过 ASCEND_RT_VISIBLE_DEVICES
# 映射为容器内逻辑 0 号卡。服务端口默认 8001（避开 DeepSeek 常用 8000），PORT 可改。
#
# 离线部署（生产机无外网时）：在有网的机器上按目标容器环境下载 whl 到 wheels/：
#   先在生产机 DeepSeek 容器里查 Python 版本和架构：
#     docker exec <容器名> python --version && docker exec <容器名> uname -m
#   再在有网机器上（--python-version/--platform 按实际值改，例：3.11 + aarch64）：
#     pip download sentence-transformers scikit-learn scipy joblib narwhals threadpoolctl \
#       -d wheels --no-deps --only-binary=:all: \
#       --python-version 311 --platform manylinux2014_aarch64 --implementation cp
#   wheels/ 放进本目录后，build 会自动走 Dockerfile.npu.offline 离线构建。
set -e
cd "$(dirname "$0")"

BASE_IMAGE="${BASE_IMAGE:-quay.io/ascend/vllm-ascend:v0.23.0}"
CARD_ID="${CARD_ID:-0}"
PORT="${PORT:-8001}"
CONTAINER=embedding-service
IMAGE=embedding-npu:latest

case "$1" in
  build)
    if [ -d "wheels" ] && ls wheels/*.whl >/dev/null 2>&1; then
      echo "检测到 wheels/ 离线包，走离线构建（无需外网）"
      docker build -f Dockerfile.npu.offline --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE" .
    else
      docker build -f Dockerfile.npu --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE" .
    fi
    ;;
  run)
    if [ ! -d "models/Qwen3-Embedding-0.6B" ]; then
      echo "错误: 缺少模型文件 models/Qwen3-Embedding-0.6B，先从本地 rsync"; exit 1
    fi
    docker rm -f "$CONTAINER" 2>/dev/null || true
    docker run -d --name "$CONTAINER" \
      --restart unless-stopped \
      --shm-size=1g --net=host --privileged=true \
      --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v "$(pwd)":/root/embedding-service \
      -e ASCEND_RT_VISIBLE_DEVICES="$CARD_ID" \
      -e EMBEDDING_DEVICE=npu \
      -e EMBEDDING_PORT="$PORT" \
      "$IMAGE" \
      python /root/embedding-service/server.py
    echo "容器已启动（卡 $CARD_ID，端口 $PORT），日志: docker logs -f $CONTAINER"
    ;;
  stop)
    docker rm -f "$CONTAINER" && echo "已停止" || echo "容器不存在"
    ;;
  status)
    curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" && echo || echo "服务未运行"
    ;;
  check)
    # 关键回归：① 装完包没动坏 torch/torch_npu 配对 ② 离线装的三个包能正常 import
    docker run --rm --privileged \
      --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
      -e ASCEND_RT_VISIBLE_DEVICES="$CARD_ID" \
      "$IMAGE" \
      python -c "import sentence_transformers, sklearn, scipy, torch, torch_npu; print('import OK | torch:', torch.__version__, '| torch_npu:', torch_npu.__version__, '| npu可用:', torch.npu.is_available())"
    ;;
  *)
    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1
    ;;
esac
