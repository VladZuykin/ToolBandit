#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_DIR="$ROOT/external/StableToolBenchQoS/server"
PROFILES_FILE="$ROOT/external/StableToolBenchQoS/data/qos/biasbusters_v1/api_qos_profiles.jsonl"
PYTHON_BIN="${QOS_PYTHON_BIN:-$ROOT/.venv-toolbench-server/Scripts/python.exe}"
LOG_DIR="${1:-$ROOT/data/benchmarks/biasbusters_cluster_comparison/qos_logs}"
PORTS=(8200 8201 8202 8203 8204 8205 8206 8207)
mkdir -p "$LOG_DIR"
# The server is started from SERVER_DIR, so redirection paths must remain
# anchored to the caller's repository rather than become relative to that cwd.
LOG_DIR="$(cd "$LOG_DIR" && pwd)"

for port in "${PORTS[@]}"; do
  if status=$(curl -fsS "http://127.0.0.1:${port}/qos/status" 2>/dev/null); then
    profile_count=$(printf '%s' "$status" | "$PYTHON_BIN" -c \
      "import json,sys; print(json.load(sys.stdin)['profiles'])")
    if [[ "$profile_count" == "50" ]]; then
      echo "QoS server already running on ${port} with 50 BiasBusters profiles"
      continue
    fi
    if [[ -f "$LOG_DIR/qos-${port}.pid" ]]; then
      old_pid=$(cat "$LOG_DIR/qos-${port}.pid")
      echo "Restarting QoS server on ${port}: it has ${profile_count} profiles instead of 50"
      kill "$old_pid" 2>/dev/null || true
      for _ in {1..20}; do
        curl -fsS "http://127.0.0.1:${port}/qos/status" >/dev/null 2>&1 || break
        sleep 0.25
      done
      if curl -fsS "http://127.0.0.1:${port}/qos/status" >/dev/null 2>&1; then
        # Git Bash signals do not always terminate a native Windows Python
        # process. The PID belongs to a server created by this script.
        taskkill.exe //PID "$old_pid" //T //F >/dev/null 2>&1 || true
        for _ in {1..20}; do
          curl -fsS "http://127.0.0.1:${port}/qos/status" >/dev/null 2>&1 || break
          sleep 0.25
        done
      fi
      if curl -fsS "http://127.0.0.1:${port}/qos/status" >/dev/null 2>&1; then
        echo "Could not stop the managed QoS server on port ${port} (PID ${old_pid})" >&2
        exit 1
      fi
    else
      echo "Port ${port} is occupied by a QoS server with ${profile_count} profiles and no managed PID" >&2
      echo "Stop that process, then rerun this script." >&2
      exit 1
    fi
  fi
  (
    cd "$SERVER_DIR"
    SERVER_PORT="$port" QOS_ENABLED=true QOS_PROFILE=normal QOS_SEED=42 \
      QOS_PROFILES_FILE="$PROFILES_FILE" QOS_SLEEP_ENABLED=false \
      "$PYTHON_BIN" main.py >"$LOG_DIR/qos-${port}.log" 2>&1 &
    echo $! >"$LOG_DIR/qos-${port}.pid"
  )
done

for port in "${PORTS[@]}"; do
  waited=0
  until curl -fsS "http://127.0.0.1:${port}/qos/status" >/dev/null 2>&1; do
    if (( waited >= 120 )); then
      echo "QoS server on port ${port} did not start; inspect $LOG_DIR/qos-${port}.log" >&2
      exit 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
done
echo "BiasBusters QoS servers are ready on ports 8200-8207"
