#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-data/benchmarks/qostoolbench_top5_clusters_originals_full_catalog}"
LIMIT="${2:-}"
DATASET="external/StableToolBenchQoS/data/benchmark/cluster_queries_v1/queries.json"
PROFILES="external/StableToolBenchQoS/data/qos/v5/api_qos_profiles.jsonl"
CATALOG="external/StableToolBenchQoS/data/catalog/tools.jsonl"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is required" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

declare -A TOOLBANDIT_PORT=(
  [agentic]=8090 [oracle-utility]=8091 [ucb-top1]=8092 [lqm-context-route]=8093
)
declare -A QOS_PORT=(
  [agentic]=8094 [oracle-utility]=8095 [ucb-top1]=8096 [lqm-context-route]=8097
)
declare -A API_SERVICE=(
  [agentic]=api-agentic [oracle-utility]=api-oracle
  [ucb-top1]=api-ucb [lqm-context-route]=api-lqm
)
POLICIES=(agentic oracle-utility ucb-top1 lqm-context-route)
TOP_CLUSTER_ARGS=(
  --cluster-id cluster_d1c71e8be4a7e9555707
  --cluster-id cluster_29db40de730e21a9528f
  --cluster-id cluster_67f7deaadfc50842a839
  --cluster-id cluster_073ac119ff0f7288b9f2
  --cluster-id cluster_62a1b14107f86a6623e2
)

run_policy() {
  local policy="$1" extra="$2"
  shift 2
  "$PYTHON_BIN" scripts/run_qostoolbench_agentic_benchmark.py \
    --dataset "$DATASET" --qos-profiles "$PROFILES" --catalog "$CATALOG" \
    --toolbandit-url "http://127.0.0.1:${TOOLBANDIT_PORT[$policy]}" \
    --toolbench-url "http://127.0.0.1:${QOS_PORT[$policy]}/virtual" \
    --output-dir "$OUTPUT_ROOT/$policy$extra" --offset 0 \
    --max-steps 6 --request-budget 0.05 --latency-sla 300 \
    --model deepseek-chat --selection-policy "$policy" \
    --candidate-scope catalog --retrieval-limit 20 --seed 42 \
    "${TOP_CLUSTER_ARGS[@]}" "$@"
}

echo "[1/4] Checking four isolated ToolBandit and QoS instances ..."
wait_for_url() {
  local label="$1" url="$2" service="${3:-}" waited=0 max_wait=900
  while ! curl -fsS "$url" >/dev/null 2>&1; do
    if (( waited >= max_wait )); then
      echo "    $label did not become ready in ${max_wait}s" >&2
      return 1
    fi
    if (( waited % 10 == 0 )); then
      progress=""
      if [[ -n "$service" ]]; then
        progress=$(docker compose -f compose.benchmark.yaml logs --no-color "$service" 2>/dev/null \
          | grep -E 'Embedded [0-9]+/|Embedding cache hit|retry [0-9]+/' | tail -n 1 || true)
      fi
      if [[ -n "$progress" ]]; then
        echo "    waiting for $label (${waited}s): ${progress#* | }"
      else
        echo "    waiting for $label (${waited}s) ..."
      fi
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "    $label ready (${waited}s)"
}
for policy in "${POLICIES[@]}"; do
  wait_for_url "ToolBandit/$policy" \
    "http://127.0.0.1:${TOOLBANDIT_PORT[$policy]}/health" "${API_SERVICE[$policy]}"
  wait_for_url "QoS/$policy" \
    "http://127.0.0.1:${QOS_PORT[$policy]}/qos/status"
  profile_count=$("$PYTHON_BIN" -c \
    "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:${QOS_PORT[$policy]}/qos/status'))['profiles'])")
  if [[ "$profile_count" != "7546" ]]; then
    echo "    QoS/$policy has $profile_count profiles; expected 7546. Restart QoS servers." >&2
    exit 1
  fi
done

echo "[2/4] Loading the 7,546-tool catalog into each in-memory index ..."
for policy in "${POLICIES[@]}"; do
  active_tools=$("$PYTHON_BIN" -c \
    "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:${TOOLBANDIT_PORT[$policy]}/health'))['active_tools'])")
  if [[ "$active_tools" == "7546" ]]; then
    echo "    [$policy] catalog already contains 7546 active tools; import skipped"
    continue
  fi
  setup_log="$OUTPUT_ROOT/$policy-catalog-setup.log"
  run_policy "$policy" "-catalog-setup" --limit 0 --reset-catalog >"$setup_log" 2>&1 &
  setup_pid=$!
  setup_started=$SECONDS
  while kill -0 "$setup_pid" 2>/dev/null; do
    elapsed=$((SECONDS - setup_started))
    embedding_progress=$(docker compose -f compose.benchmark.yaml logs --no-color "${API_SERVICE[$policy]}" 2>/dev/null \
      | grep -E 'Embedded [0-9]+/|Embedding cache hit' | tail -n 1 || true)
    if [[ -n "$embedding_progress" ]]; then
      echo "    [$policy] elapsed=${elapsed}s ${embedding_progress#* | }"
    else
      echo "    [$policy] elapsed=${elapsed}s preparing/importing catalog; waiting for embedding progress ..."
    fi
    sleep 10
  done
  if ! wait "$setup_pid"; then
    tail -n 40 "$setup_log" >&2
    exit 1
  fi
  echo "    [$policy] catalog ready in $((SECONDS - setup_started))s"
done

echo "[3/4] Resetting QoS sequences and starting four policies in parallel ..."
declare -A RUN_PID=()
for policy in "${POLICIES[@]}"; do
  if [[ -f "$OUTPUT_ROOT/$policy/summary.json" ]] && \
     "$PYTHON_BIN" -c \
       "import json,sys; sys.exit(0 if json.load(open(sys.argv[1], encoding='utf-8')).get('qos_server_profiles') == 7546 else 1)" \
       "$OUTPUT_ROOT/$policy/summary.json"; then
    echo "    [$policy] completed summary found; reusing it"
    continue
  fi
  LIMIT_ARGS=()
  if [[ -n "$LIMIT" ]]; then LIMIT_ARGS=(--limit "$LIMIT"); fi
  if [[ -s "$OUTPUT_ROOT/$policy/trajectories.jsonl" ]]; then
    failed_suffix=$(date '+%Y%m%d-%H%M%S')
    failed_dir="$OUTPUT_ROOT/$policy-failed-$failed_suffix"
    mv "$OUTPUT_ROOT/$policy" "$failed_dir"
    if [[ -f "$OUTPUT_ROOT/$policy.log" ]]; then
      mv "$OUTPUT_ROOT/$policy.log" "$failed_dir.log"
    fi
    echo "    [$policy] archived interrupted run; resetting in-memory learning"
    docker compose -f compose.benchmark.yaml restart "${API_SERVICE[$policy]}" >/dev/null
    wait_for_url "ToolBandit/$policy" \
      "http://127.0.0.1:${TOOLBANDIT_PORT[$policy]}/health" "${API_SERVICE[$policy]}"
  fi
  curl -fsS -X POST "http://127.0.0.1:${QOS_PORT[$policy]}/qos/reset" >/dev/null
  run_policy "$policy" "" "${LIMIT_ARGS[@]}" >"$OUTPUT_ROOT/$policy.log" 2>&1 &
  RUN_PID[$policy]=$!
done

status=0
echo "Following per-policy progress (ETA stabilizes after several tasks) ..."
while true; do
  running=0
  echo "--- $(date '+%H:%M:%S') ---"
  for policy in "${POLICIES[@]}"; do
    if [[ -n "${RUN_PID[$policy]:-}" ]] && kill -0 "${RUN_PID[$policy]}" 2>/dev/null; then
      running=1
    fi
    printf '%-20s ' "$policy"
    if [[ -f "$OUTPUT_ROOT/$policy.log" ]]; then
      tail -n 1 "$OUTPUT_ROOT/$policy.log"
    else
      echo "waiting for log"
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
  echo "At least one run failed; inspect $OUTPUT_ROOT/*.log" >&2
  exit "$status"
fi

echo "[4/4] Building comparison ..."
"$PYTHON_BIN" scripts/compare_qostoolbench_agentic_runs.py \
  "$OUTPUT_ROOT/agentic/summary.json" \
  "$OUTPUT_ROOT/oracle-utility/summary.json" \
  "$OUTPUT_ROOT/ucb-top1/summary.json" \
  "$OUTPUT_ROOT/lqm-context-route/summary.json" \
  --json-output "$OUTPUT_ROOT/comparison.json" \
  --markdown-output "$OUTPUT_ROOT/comparison.md"
