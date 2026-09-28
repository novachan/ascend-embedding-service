#!/usr/bin/env bash
# 一键离线安装 + 启动 embedding 服务（在生产 DeepSeek 容器内执行）
#
# 用法:
#   CARD_ID=5 ./install-and-start.sh          # 安装缺失依赖并启动（默认）
#   ./install-and-start.sh stop               # 停止
#   ./install-and-start.sh status             # 状态
#
# CARD_ID = npu-smi info 里剩余显存最多的卡号（容器内执行 npu-smi info 查看）
# PORT    = 服务端口，默认 8001
#
# 安全保证：pip 用 --no-index --find-links=wheels 离线安装，只补缺失的包
# （sentence-transformers/scikit-learn/joblib/narwhals/threadpoolctl），
# 容器已有的 torch/torch_npu/transformers/scipy 等一律不升级。
# 注意：容器若被重建，此安装会丢失，重新执行本脚本即可（幂等）。
set -e
cd "$(dirname "$0")"

PORT="${PORT:-8001}"
CARD_ID="${CARD_ID:-0}"
PID_FILE=server.pid

health() { curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null; }

case "${1:-start}" in
  start)
    if health >/dev/null; then
      echo "服务已在运行: $(health)"; exit 0
    fi
    [ -d "wheels" ] && ls wheels/*.whl >/dev/null 2>&1 || { echo "错误: 缺少 wheels/ 离线包"; exit 1; }
    [ -d "models/Qwen3-Embedding-0.6B" ] || { echo "错误: 缺少 models/Qwen3-Embedding-0.6B"; exit 1; }

    echo "==> [1/3] 离线安装缺失依赖（只装缺的，已有的绝不升级）"
    python -m pip install --no-index --find-links=wheels sentence-transformers scikit-learn

    echo "==> [2/3] 验证 torch_npu 配对"
    python -c "import sentence_transformers, sklearn, torch, torch_npu; print('import OK | torch:', torch.__version__, '| torch_npu:', torch_npu.__version__, '| npu可用:', torch.npu.is_available())"

    echo "==> [3/3] 启动服务（卡 $CARD_ID，端口 $PORT），首次加载模型约 15-30 秒"
    EMBEDDING_DEVICE=npu EMBEDDING_PORT="$PORT" ASCEND_RT_VISIBLE_DEVICES="$CARD_ID" \
      nohup python server.py > server.log 2>&1 &
    echo $! > "$PID_FILE"
    echo -n "等待模型加载"
    for _ in $(seq 1 30); do
      sleep 2; echo -n "."
      if health >/dev/null; then
        echo; echo "✅ 服务就绪: $(health)"
        echo "   验证: curl -s http://127.0.0.1:${PORT}/v1/models"
        echo "   日志: tail -f $(pwd)/server.log"
        exit 0
      fi
    done
    echo; echo "❌ 60 秒未就绪，排查: tail -50 $(pwd)/server.log"; exit 1
    ;;
  stop)
    if [ -f "$PID_FILE" ]; then
      kill "$(cat "$PID_FILE")" && rm -f "$PID_FILE" && echo "已停止"
    else
      echo "无 PID 文件（可能未启动或容器已重建）"
    fi
    ;;
  status)
    health && echo || echo "服务未运行"
    ;;
  *)
    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1
    ;;
esac
