#!/usr/bin/env bash
# 启动 embedding 服务容器：基于 vllm-ascend 官方镜像（CANN/torch_npu 环境现成且版本配对），
# 不跑 vLLM 引擎，容器内用 SentenceTransformer 借用 NPU 空闲显存推理。
#
# 用法（宿主机）:
#   ./docker-create.sh                        # 默认 HOST_PORT=8090，DATA_DIR=/data/qwen3-embeding
#   HOST_PORT=8091 DATA_DIR=/data/embed ./docker-create.sh
#
# 说明:
#   - davinci0-7 为 NPU 设备，按宿主机实际卡数与空闲情况裁剪
#   - --shm-size 500g 沿用生产实配，可按宿主机内存调小
#   - DATA_DIR 内放 embedding-service/（本仓库内容）与模型，挂载为容器 /root/
set -e

HOST_PORT="${HOST_PORT:-8090}"
DATA_DIR="${DATA_DIR:-/data/qwen3-embeding}"

docker run -itd --privileged --name=qwen3-embeding-vllm20 -p ${HOST_PORT}:8001 \
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
  -v ${DATA_DIR}/:/root/ \
  -w /root/ \
  quay.io/ascend/vllm-ascend:v0.20.2rc1 \
  bash

echo "容器已启动，进入容器: docker exec -it qwen3-embeding-vllm20 bash"
