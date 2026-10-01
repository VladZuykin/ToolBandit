#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_qostoolbench_comparison.sh [limit] [output-dir] [policies] [cluster-id] [source-query-id] [dataset] [candidate-scope] [retrieval-limit]
LIMIT="${1:-193}"
OUTPUT_ROOT="${2:-data/benchmarks/qostoolbench_comparison}"
POLICY_LIST="${3:-agentic,ucb-top1,semantic,cheapest,fastest}"
CLUSTER_ID="${4:-}"
SOURCE_QUERY_ID="${5:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${6:-external/StableToolBenchQoS/data/benchmark/cluster_queries_v1/queries.json}"
CANDIDATE_SCOPE="${7:-annotated}"
RETRIEVAL_LIMIT="${8:-20}"
PROFILES="external/StableToolBenchQoS/data/qos/v5/api_qos_profiles.jsonl"
IFS=',' read -r -a POLICIES <<< "$POLICY_LIST"
CLUSTER_ARGS=()
if [[ -n "$CLUSTER_ID" ]]; then
  CLUSTER_ARGS=(--cluster-id "$CLUSTER_ID")
fi
SOURCE_QUERY_ARGS=()
if [[ -n "$SOURCE_QUERY_ID" ]]; then
  SOURCE_QUERY_ARGS=(--source-query-id "$SOURCE_QUERY_ID")
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is required" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

for policy in "${POLICIES[@]}"; do
  echo "=== ${policy}: restarting ToolBandit to reset in-memory learning ==="
  docker compose restart api
  until curl -fsS http://127.0.0.1:8080/health >/dev/null; do
    sleep 2
  done

  "$PYTHON_BIN" scripts/run_qostoolbench_agentic_benchmark.py \
    --dataset "$DATASET" \
    --qos-profiles "$PROFILES" \
    --toolbandit-url http://127.0.0.1:8080 \
    --toolbench-url http://127.0.0.1:8082/virtual \
    --output-dir "$OUTPUT_ROOT/$policy" \
    --offset 0 \
    --limit "$LIMIT" \
    --max-steps 6 \
    --request-budget 0.05 \
    --latency-sla 300 \
    --model deepseek-chat \
    --selection-policy "$policy" \
    --candidate-scope "$CANDIDATE_SCOPE" \
    --retrieval-limit "$RETRIEVAL_LIMIT" \
    --seed 42 \
    "${CLUSTER_ARGS[@]}" \
    "${SOURCE_QUERY_ARGS[@]}" \
    --reset-qos
done

"$PYTHON_BIN" scripts/compare_qostoolbench_agentic_runs.py \
  "$OUTPUT_ROOT"/*/summary.json \
  --json-output "$OUTPUT_ROOT/comparison.json" \
  --markdown-output "$OUTPUT_ROOT/comparison.md"

echo "Saved comparison to $OUTPUT_ROOT/comparison.md and comparison.json"
