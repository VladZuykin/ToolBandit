#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_DIR="$ROOT/external/StableToolBenchQoS/server"
PROFILES_FILE="$ROOT/external/StableToolBenchQoS/data/qos/v5/api_qos_profiles.jsonl"
PYTHON_BIN="${QOS_PYTHON_BIN:-$ROOT/.venv-toolbench-server/Scripts/python.exe}"
LOG_DIR="${1:-$ROOT/data/benchmarks/qos_server_logs}"
mkdir -p "$LOG_DIR"

for port in 8094 8095 8096 8097; do
  if curl -fsS "http://127.0.0.1:${port}/docs" >/dev/null 2>&1; then
    echo "QoS server already running on ${port}"
    continue
  fi
  (
    cd "$SERVER_DIR"
    SERVER_PORT="$port" QOS_ENABLED=true QOS_PROFILE=normal QOS_SEED=42 \
      QOS_PROFILES_FILE="$PROFILES_FILE" \
      QOS_SLEEP_ENABLED=false "$PYTHON_BIN" main.py \
      >"$LOG_DIR/qos-${port}.log" 2>&1 &
    echo $! >"$LOG_DIR/qos-${port}.pid"
  )
done

for port in 8094 8095 8096 8097; do
  until curl -fsS "http://127.0.0.1:${port}/docs" >/dev/null; do sleep 1; done
done
echo "QoS benchmark servers are ready on ports 8094-8097 with $PROFILES_FILE"
