#!/usr/bin/env bash
set -euo pipefail

SAMPLES_PER_CLUSTER="${1:-10}"
OUTPUT_ROOT="${2:-data/benchmarks/biasbusters_cluster_comparison}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="../BiasBusters/3_generate_queries_for_clusters/cluster_queries.json"
CLUSTERS="../BiasBusters/2_generate_clusters_and_refine/duplicate_api_clusters.json"
PROFILES="external/StableToolBenchQoS/data/qos/biasbusters_v1/api_qos_profiles.jsonl"
COMPOSE_FILE="compose.biasbusters.yaml"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is required" >&2
  exit 1
fi
if [[ ! -f "$DATASET" || ! -f "$CLUSTERS" || ! -f "$PROFILES" ]]; then
  echo "BiasBusters dataset, clusters, or QoS profiles are missing" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

POLICIES=(
  ucb-top1 lqm-context-route semantic random
  cheapest fastest highest-pass-rate oracle-utility
)
declare -A TB_PORT=(
  [ucb-top1]=8100 [lqm-context-route]=8101 [semantic]=8102 [random]=8103
  [cheapest]=8104 [fastest]=8105 [highest-pass-rate]=8106 [oracle-utility]=8107
)
declare -A QOS_PORT=(
  [ucb-top1]=8200 [lqm-context-route]=8201 [semantic]=8202 [random]=8203
  [cheapest]=8204 [fastest]=8205 [highest-pass-rate]=8206 [oracle-utility]=8207
)
declare -A API_SERVICE=(
  [ucb-top1]=api-ucb [lqm-context-route]=api-lqm [semantic]=api-semantic
  [random]=api-random [cheapest]=api-cheapest [fastest]=api-fastest
  [highest-pass-rate]=api-passrate [oracle-utility]=api-oracle
)

echo "[0/5] Verifying OpenAI and DeepSeek credentials before the parallel run ..."
if ! "$PYTHON_BIN" scripts/check_api_connections.py; then
  echo "API credential preflight failed; no benchmark strategies were started." >&2
  exit 1
fi

wait_for_url() {
  local label="$1" url="$2" waited=0
  until curl -fsS "$url" >/dev/null 2>&1; do
    if (( waited >= 900 )); then
      echo "$label did not become ready in 900s" >&2
      return 1
    fi
    if (( waited % 10 == 0 )); then echo "    waiting for $label (${waited}s) ..."; fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "    $label ready (${waited}s)"
}

run_policy() {
  local policy="$1" extra="$2"
  shift 2
  "$PYTHON_BIN" scripts/run_biasbusters_online_benchmark.py \
    --dataset "$DATASET" --clusters "$CLUSTERS" --qos-profiles "$PROFILES" \
    --toolbandit-url "http://127.0.0.1:${TB_PORT[$policy]}" \
    --toolbench-url "http://127.0.0.1:${QOS_PORT[$policy]}/virtual" \
    --output "$OUTPUT_ROOT/$policy$extra/trajectories.jsonl" \
    --summary "$OUTPUT_ROOT/$policy$extra/summary.json" \
    --samples-per-cluster "$SAMPLES_PER_CLUSTER" --seed 42 \
    --max-attempts 3 --request-budget 0.05 --latency-sla 300 \
    --retrieval-limit 5 --selection-policy "$policy" \
    --argument-generator deepseek --argument-model deepseek-chat "$@"
}

echo "[1/5] Starting isolated ToolBandit and QoS instances ..."
docker compose -f "$COMPOSE_FILE" up -d --build --force-recreate
bash scripts/start_biasbusters_qos_servers.sh "$OUTPUT_ROOT/qos_logs"
for policy in "${POLICIES[@]}"; do
  wait_for_url "ToolBandit/$policy" "http://127.0.0.1:${TB_PORT[$policy]}/health"
  wait_for_url "QoS/$policy" "http://127.0.0.1:${QOS_PORT[$policy]}/qos/status"
  profile_count=$(curl -fsS "http://127.0.0.1:${QOS_PORT[$policy]}/qos/status" | \
    "$PYTHON_BIN" -c "import json,sys; print(json.load(sys.stdin)['profiles'])")
  if [[ "$profile_count" != "50" ]]; then
    echo "QoS/$policy has $profile_count profiles; expected 50 BiasBusters profiles" >&2
    exit 1
  fi
done

echo "[2/5] Preparing the same 50-tool clustered catalog for every strategy ..."
for policy in "${POLICIES[@]}"; do
  active_tools=$("$PYTHON_BIN" -c \
    "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:${TB_PORT[$policy]}/health'))['active_tools'])")
  if [[ "$active_tools" == "50" ]]; then
    echo "    [$policy] 50 tools already loaded"
    continue
  fi
  setup_log="$OUTPUT_ROOT/$policy-catalog-setup.log"
  run_policy "$policy" "-catalog-setup" --limit 0 --reset-catalog >"$setup_log" 2>&1 || {
    tail -n 60 "$setup_log" >&2
    exit 1
  }
  echo "    [$policy] catalog loaded"
done

echo "[3/5] Resetting learning and deterministic QoS sequences ..."
declare -A REUSE=()
for policy in "${POLICIES[@]}"; do
  summary="$OUTPUT_ROOT/$policy/summary.json"
  if [[ -f "$summary" ]] && "$PYTHON_BIN" -c \
    "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); n=int(sys.argv[2]); sys.exit(0 if d.get('selection_policy')==sys.argv[3] and d.get('samples_per_cluster')==n and d.get('queries')==n*10 else 1)" \
    "$summary" "$SAMPLES_PER_CLUSTER" "$policy"; then
    REUSE[$policy]=1
    echo "    [$policy] completed result found; reusing it"
    continue
  fi
  if [[ -d "$OUTPUT_ROOT/$policy" ]]; then
    suffix=$(date '+%Y%m%d-%H%M%S')
    mv "$OUTPUT_ROOT/$policy" "$OUTPUT_ROOT/$policy-previous-$suffix"
  fi
  docker compose -f "$COMPOSE_FILE" restart "${API_SERVICE[$policy]}" >/dev/null
done
for policy in "${POLICIES[@]}"; do
  [[ -n "${REUSE[$policy]:-}" ]] && continue
  wait_for_url "ToolBandit/$policy" "http://127.0.0.1:${TB_PORT[$policy]}/health"
  curl -fsS -X POST "http://127.0.0.1:${QOS_PORT[$policy]}/qos/reset" >/dev/null
done

echo "[4/5] Running eight strategies in parallel ($SAMPLES_PER_CLUSTER queries per cluster) ..."
declare -A RUN_PID=()
for policy in "${POLICIES[@]}"; do
  [[ -n "${REUSE[$policy]:-}" ]] && continue
  mkdir -p "$OUTPUT_ROOT/$policy"
  run_policy "$policy" "" --skip-import >"$OUTPUT_ROOT/$policy.log" 2>&1 &
  RUN_PID[$policy]=$!
done

status=0
while true; do
  running=0
  echo "--- $(date '+%H:%M:%S') ---"
  for policy in "${POLICIES[@]}"; do
    if [[ -n "${RUN_PID[$policy]:-}" ]] && kill -0 "${RUN_PID[$policy]}" 2>/dev/null; then
      running=1
    fi
    printf '%-22s ' "$policy"
    if [[ -n "${REUSE[$policy]:-}" ]]; then
      echo "completed result reused"
    elif [[ -f "$OUTPUT_ROOT/$policy.log" ]]; then
      tail -n 1 "$OUTPUT_ROOT/$policy.log"
    else
      echo "waiting"
    fi
  done
  [[ "$running" -eq 0 ]] && break
  sleep 30
done
for policy in "${POLICIES[@]}"; do
  if [[ -n "${RUN_PID[$policy]:-}" ]]; then
    wait "${RUN_PID[$policy]}" || status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one strategy failed; inspect $OUTPUT_ROOT/*.log" >&2
  exit "$status"
fi

echo "[5/5] Building aggregate comparison ..."
summaries=()
for policy in "${POLICIES[@]}"; do summaries+=("$OUTPUT_ROOT/$policy/summary.json"); done
"$PYTHON_BIN" scripts/compare_biasbusters_runs.py "${summaries[@]}" \
  --json-output "$OUTPUT_ROOT/comparison.json" \
  --markdown-output "$OUTPUT_ROOT/comparison.md"
