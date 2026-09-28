#!/usr/bin/env bash
# Embedding 服务管理脚本：start / stop / status
cd "$(dirname "$0")"

case "$1" in
  start)
    if curl -sf --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
      echo "服务已在运行"
      exit 0
    fi
    nohup .venv/bin/python server.py > server.log 2>&1 &
    echo "启动中，PID: $!（首次加载模型约 15-30 秒）"
    ;;
  stop)
    pkill -f "python server.py" && echo "已停止" || echo "服务未在运行"
    ;;
  status)
    curl -s --max-time 2 http://127.0.0.1:8000/health && echo || echo "服务未运行"
    ;;
  *)
    echo "用法: $0 {start|stop|status}"
    exit 1
    ;;
esac
