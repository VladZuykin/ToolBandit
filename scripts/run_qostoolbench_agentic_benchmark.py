"""Run QoSToolBench cluster tasks with an LLM choosing tools step by step."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import re
import random
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid


BASE_SCRIPT = Path(__file__).with_name("run_biasbusters_online_benchmark.py")
SPEC = importlib.util.spec_from_file_location("biasbusters_online", BASE_SCRIPT)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

AGENT_PROMPT_VERSION = "qostoolbench-agent-selector-v2"
AGENT_SELECTION_INSTRUCTIONS = """You are a tool-selection agent. Solve USER_TASK using only the top-20
AVAILABLE_TOOLS returned by semantic retrieval.

Selection rules:
1. Functional compatibility is mandatory: first inspect description and input_schema.
2. retrieval_score measures semantic similarity, not observed reliability.
3. ucb_score is ToolBandit's optimistic online utility estimate. A high value can mean
   strong observed outcomes or uncertainty worth exploring; it is evidence, not proof.
4. lqm_score estimates query-conditional quality per service cycle and deflates
   exploration of candidates whose predicted quality is dominated.
5. expected_cost and expected_latency are current estimates. Prefer a reliable,
   functionally correct tool while avoiding unnecessary cost and latency.
6. Use PREVIOUS_STEPS: do not repeat a failed tool, and adapt arguments or tool choice
   to errors and incomplete outputs. Multiple calls are allowed only when needed.
7. Finish only when tool evidence materially satisfies the task. Never invent results."""


def llm_json(prompt: str, *, model: str, timeout: float, retries: int) -> dict[str, Any]:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return one strict JSON object only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(retries + 1):
        request = Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return BASE._json_object(body["choices"][0]["message"]["content"])
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError("DeepSeek HTTP {}: {}".format(error.code, detail)) from error
        except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
            if attempt >= retries:
                raise RuntimeError("DeepSeek unavailable: {}".format(error)) from error
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            if attempt >= retries:
                raise RuntimeError(
                    "DeepSeek returned invalid JSON after {} attempts: {}".format(
                        attempt + 1, error
                    )
                ) from error
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("DeepSeek returned no result")


def planner_prompt(query: str, candidates: list[dict[str, Any]], history: list[dict[str, Any]]) -> str:
    tools = [{
        "tool": item["tool"], "ucb_score": item["ucb_score"],
        "lqm_score": item.get("lqm_score"),
        "retrieval_score": item["retrieval_score"],
        "expected_cost": item.get("expected_cost"),
        "expected_latency": item.get("expected_latency"),
        "input_schema": item.get("input_schema", {}),
        "description": (item.get("metadata") or {}).get("api_description", ""),
    } for item in candidates]
    compact_history = [{
        "tool": step["tool"], "arguments": step["arguments"],
        "output": BASE.preview(step.get("tool_response"), 1800),
        "error": step.get("tool_error"), "judge_verdict": step.get("judge_verdict"),
    } for step in history]
    return """{}

If another call is needed, return:
{{"action":"call","tool":"exact tool name","arguments":{{}},"reason":"short reason"}}
If the task is solved, return:
{{"action":"finish","answer":"final answer for the user","reason":"short reason"}}

USER_TASK:
{}

PREVIOUS_STEPS:
{}

AVAILABLE_TOOLS:
{}""".format(
        AGENT_SELECTION_INSTRUCTIONS, query, json.dumps(compact_history, ensure_ascii=False),
        json.dumps(tools, ensure_ascii=False),
    )


def final_judge_prompt(query: str, answer: str, history: list[dict[str, Any]]) -> str:
    evidence = [{
        "tool": step["tool"], "output": BASE.preview(step.get("tool_response"), 2500),
        "error": step.get("tool_error"),
    } for step in history]
    return """Evaluate whether FINAL_ANSWER actually delivers the result requested in USER_TASK
using the tool evidence. A truthful apology, report of tool failure, description of limitations,
or instructions for how the user could do it later does NOT complete the task and must be fail.
Use partial only when a useful requested deliverable was produced but some requirements are missing.
Use pass only when all material requested outputs were actually produced. Do not reward unsupported
claims. Return exactly:
{{"passed":true,"verdict":"pass|partial|fail","confidence":0.0,
 "reason_code":"CODE","explanation":"short explanation","missing_requirements":[]}}

USER_TASK:
{}
TOOL_EVIDENCE:
{}
FINAL_ANSWER:
{}""".format(query, json.dumps(evidence, ensure_ascii=False), answer)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "unknown"


def load_catalog_apis(path: Path) -> dict[str, dict[str, Any]]:
    """Load the complete StableToolBenchQoS catalog in benchmark API shape."""
    by_api_id: dict[str, dict[str, Any]] = {}

    def add_api(api: dict[str, Any]) -> None:
        identity = str(api.get("_api_id") or BASE.api_id(api))
        api["_api_id"] = identity
        by_api_id.setdefault(identity, api)

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            api = {
                "_api_id": record.get("api_id", ""),
                "category_name": record.get("category", ""),
                "tool_name": record.get("tool_name", ""),
                "api_name": record.get("api_name", ""),
                "api_description": record.get("api_description", ""),
                "required_parameters": record.get("required_parameters", []),
                "optional_parameters": record.get("optional_parameters", []),
                "method": record.get("method", ""),
            }
            if not api["tool_name"] or not api["api_name"]:
                raise ValueError("Invalid catalog entry at {}:{}".format(path, line_number))
            add_api(api)

    # A small number of solvable-query APIs are intentionally retained in the
    # QoS target set although their definitions are absent from tools.jsonl.
    # Recover those real definitions from the benchmark sources rather than
    # creating description-less synthetic tools.
    repository = path.resolve().parents[2]
    for root_name in ("solvable_queries", "solvable_queries_example"):
        root = repository / root_name
        if not root.is_dir():
            continue
        for query_file in root.rglob("*.json"):
            payload = json.loads(query_file.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                for source_api in row.get("api_list", []):
                    api = dict(source_api)
                    api["_api_id"] = BASE.api_id(api)
                    add_api(api)

    result: dict[str, dict[str, Any]] = {}
    for identity, api in by_api_id.items():
        # ToolBench permits identical display names in different categories;
        # exact api_id makes every QoS profile a distinct bandit arm.
        name = identity
        if len(name) > 255:
            suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            name = name[:238] + "_" + suffix
        result[name] = api
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--qos-profiles", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Complete tools.jsonl catalog. When omitted, tools are read from dataset api_list values.",
    )
    parser.add_argument("--toolbandit-url", default="http://127.0.0.1:8080")
    parser.add_argument("--toolbench-url", default="http://127.0.0.1:8082/virtual")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--cluster-id", action="append",
        help="Run tasks belonging to this cluster; repeat to select a union of clusters.",
    )
    parser.add_argument(
        "--source-query-id",
        help="Run only paraphrases derived from this source query ID.",
    )
    parser.add_argument(
        "--candidate-scope", choices=("annotated", "catalog"), default="annotated",
        help="Restrict retrieval to annotated alternatives or search the complete catalog.",
    )
    parser.add_argument("--retrieval-limit", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--request-budget", type=float, default=0.05)
    parser.add_argument("--latency-sla", type=float, default=300.0)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument(
        "--selection-policy",
        choices=("agentic", "ucb-top1", "semantic", "random", "cheapest",
                 "fastest", "highest-pass-rate", "oracle-utility", "lqm-context-route"),
        default="agentic",
        help="Who selects the next tool. agentic lets the LLM choose; baselines force a policy choice.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oracle-cost-weight", type=float, default=0.5)
    parser.add_argument("--oracle-latency-weight", type=float, default=0.5)
    parser.add_argument("--llm-timeout", type=float, default=90.0)
    parser.add_argument("--llm-retries", type=int, default=4)
    parser.add_argument("--judge-timeout", type=float, default=180.0)
    parser.add_argument("--reset-catalog", action="store_true")
    parser.add_argument("--reset-qos", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Append to an interrupted run and skip query IDs already in trajectories.jsonl.",
    )
    args = parser.parse_args()

    all_rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    eligible_rows = all_rows
    if args.cluster_id:
        selected_clusters = set(args.cluster_id)
        eligible_rows = [
            row for row in all_rows
            if selected_clusters.intersection(
                (row.get("cluster_benchmark") or {}).get("cluster_ids") or []
            )
        ]
        if not eligible_rows:
            parser.error("cluster-id values were not found in the dataset: {}".format(
                ", ".join(args.cluster_id)
            ))
    if args.source_query_id:
        eligible_rows = [
            row for row in eligible_rows
            if str((row.get("paraphrase") or {}).get("source_query_id", row.get("query_id")))
            == str(args.source_query_id)
        ]
        if not eligible_rows:
            parser.error("source-query-id was not found after filtering: {}".format(
                args.source_query_id
            ))
    rows = eligible_rows[args.offset:]
    if args.limit is not None:
        rows = rows[:args.limit]
    profiles = BASE.load_profiles(args.qos_profiles, "normal")
    if args.catalog:
        catalog_apis = load_catalog_apis(args.catalog)
        apis = {
            name: api for name, api in catalog_apis.items()
            if BASE.api_id(api) in profiles
        }
    else:
        apis = BASE.unique_apis(all_rows)
    definitions = BASE.catalog_definitions(apis, profiles)
    base = args.toolbandit_url.rstrip("/")
    health = BASE.request_json(base + "/health")
    selection_rng = random.Random(args.seed)
    print("[1/5] Services: ToolBandit={} tools={}".format(health["status"], health["active_tools"]), flush=True)
    with urlopen(args.toolbench_url.rsplit("/", 1)[0] + "/docs", timeout=10):
        pass
    qos_status = BASE.request_json(args.toolbench_url.rsplit("/", 1)[0] + "/qos/status")
    if not qos_status.get("enabled") or int(qos_status.get("profiles") or 0) != len(profiles):
        raise RuntimeError(
            "QoS server profile mismatch: expected {} enabled profiles, got {}".format(
                len(profiles), qos_status
            )
        )
    if args.reset_qos:
        BASE.request_json(args.toolbench_url.rsplit("/", 1)[0] + "/qos/reset", method="POST")
    if args.reset_catalog:
        existing = BASE.request_json(base + "/v1/tools?limit=1000").get("items", [])
        # Do not delete tools one by one: every registry mutation rebuilds the
        # semantic index, which causes unnecessary embedding calls. Upsert the
        # benchmark catalog once; candidate_tools still restricts every task to
        # its annotated cluster APIs.
        print("[2/5] Upserting {} tools over {} existing ...".format(len(definitions), len(existing)), flush=True)
        BASE.request_json(base + "/v1/tools/import", method="POST",
                          payload={"tools": definitions, "replace_existing": True}, timeout=1800)
    else:
        print("[2/5] Catalog import skipped", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = args.output_dir / "trajectories.jsonl"
    run_id = uuid.uuid4().hex
    existing_records: list[dict[str, Any]] = []
    if args.resume and trajectories_path.exists():
        for line in trajectories_path.read_text(encoding="utf-8").splitlines():
            try:
                existing_records.append(json.loads(line))
            except json.JSONDecodeError:
                print("Ignoring an incomplete trailing checkpoint line", flush=True)
    completed_ids = {str(record["query_id"]) for record in existing_records}
    pending_rows = [row for row in rows if str(row["query_id"]) not in completed_ids]
    successes = sum(int(record.get("passed") is True) for record in existing_records)
    total_steps = sum(len(record.get("steps") or []) for record in existing_records)
    total_cost = sum(
        float(step.get("cost") or 0.0)
        for record in existing_records for step in (record.get("steps") or [])
    )
    initialized_cluster_files: set[str] = {
        str(cluster_id)
        for record in existing_records for cluster_id in (record.get("cluster_ids") or [])
    }
    cluster_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"queries": 0, "successes": 0, "steps": 0, "cost": 0.0})
    for record in existing_records:
        for cluster_id in record.get("cluster_ids") or ["unassigned"]:
            stat = cluster_stats[str(cluster_id)]
            steps = record.get("steps") or []
            stat["queries"] += 1
            stat["successes"] += int(record.get("passed") is True)
            stat["steps"] += len(steps)
            stat["cost"] += sum(float(step.get("cost") or 0.0) for step in steps)
    print(
        "[3/5] Running {} agent tasks ({} checkpointed, {} remaining) ...".format(
            len(rows), len(existing_records), len(pending_rows)
        ),
        flush=True,
    )
    benchmark_started = time.monotonic()
    stream_mode = "a" if args.resume and trajectories_path.exists() else "w"
    with trajectories_path.open(stream_mode, encoding="utf-8", newline="\n") as stream:
        for pending_position, row in enumerate(pending_rows, start=1):
            position = len(existing_records) + pending_position
            query = str(row["query"])
            candidate_names = [BASE.tool_name(api) for api in row["api_list"]]
            clusters = list((row.get("cluster_benchmark") or {}).get("cluster_ids") or ["unassigned"])
            remaining = args.request_budget
            excluded: list[str] = []
            history: list[dict[str, Any]] = []
            final_answer = ""
            planner_error = None
            for step_index in range(1, args.max_steps + 1):
                context = query + "\n" + json.dumps(history, ensure_ascii=False)[-6000:]
                retrieval_limit = (len(candidate_names) if args.candidate_scope == "annotated"
                                   else args.retrieval_limit)
                search_payload = {
                    "query": context, "remaining_budget": remaining,
                    "latency_sla": args.latency_sla,
                    "retrieval_limit": retrieval_limit,
                    "result_limit": retrieval_limit,
                    "retrieval_threshold": -1.0,
                    "excluded_tools": excluded,
                }
                if args.candidate_scope == "annotated":
                    search_payload["candidate_tools"] = candidate_names
                search = BASE.request_json(
                    base + "/v1/tools/search", method="POST", retries=4,
                    payload=search_payload,
                )
                candidates = search.get("tools", [])
                if not candidates:
                    planner_error = "no feasible candidates"
                    break
                for item in candidates:
                    profile = BASE.profile_for_candidate(item, profiles)
                    item["expected_cost"] = profile["cost"]
                    item["expected_latency"] = profile["latency"]
                visible_candidates = candidates
                forced_candidate = None
                if args.selection_policy != "agentic":
                    forced_candidate = BASE.select_candidate(
                        candidates, policy=args.selection_policy, profiles=profiles,
                        rng=selection_rng,
                        cost_weight=args.oracle_cost_weight,
                        latency_weight=args.oracle_latency_weight,
                        cost_scale=float(health.get("cost_scale", 0.012)),
                        latency_scale=float(health.get("latency_scale", 3.0)),
                    )
                    visible_candidates = [forced_candidate]
                decision = llm_json(planner_prompt(query, visible_candidates, history), model=args.model,
                                    timeout=args.llm_timeout, retries=args.llm_retries)
                if decision.get("action") == "finish" and not history:
                    decision = llm_json(
                        planner_prompt(query, visible_candidates, history)
                        + "\n\nNo tool evidence exists yet. You MUST return action=call and "
                          "arguments for one AVAILABLE_TOOL; action=finish is invalid.",
                        model=args.model,
                        timeout=args.llm_timeout,
                        retries=args.llm_retries,
                    )
                    if decision.get("action") != "call":
                        planner_error = "LLM attempted to finish without tool evidence"
                        break
                if decision.get("action") == "finish" and history:
                    final_answer = str(decision.get("answer") or "")
                    break
                name = (str(forced_candidate["tool"]) if forced_candidate is not None
                        else str(decision.get("tool") or ""))
                candidate = next((item for item in candidates if item["tool"] == name), None)
                if candidate is None:
                    planner_error = "LLM selected unavailable tool: " + name
                    break
                metadata = dict(candidate.get("metadata") or {})
                arguments = decision.get("arguments")
                if not isinstance(arguments, dict):
                    planner_error = "LLM arguments are not an object"
                    break
                call = BASE.request_json(args.toolbench_url, method="POST", timeout=180,
                    payload={"category": metadata["category_name"], "tool_name": metadata["tool_name"],
                             "api_name": metadata["api_name"], "tool_input": arguments,
                             "strip": "", "toolbench_key": ""})
                qos = dict(call.get("qos") or {})
                cost = float(qos.get("cost_units") or 0.0)
                latency = float(qos.get("latency_ms") or 0.0) / 1000.0
                evaluation = BASE.request_json(base + "/v1/evaluations", method="POST", retries=4, payload={
                    "interaction_id": "qos-agent-{}-{}-{}".format(run_id, row["query_id"], step_index),
                    "request_id": search["request_id"], "tool_name": name,
                    "tool_intent": query, "expected_contract": BASE.expected_contract(metadata),
                    "tool_output": call, "observed_cost": cost, "observed_latency": latency})
                job = BASE.wait_for_judge(base, str(evaluation["evaluation_id"]), args.judge_timeout, 1.5)
                judgment = dict(job.get("judgment") or {})
                step = {"step": step_index, "reason": decision.get("reason"), "tool": name,
                        "arguments": arguments, "ucb_score": candidate.get("ucb_score"),
                        "lqm_score": candidate.get("lqm_score"),
                        "retrieval_rank": candidate.get("retrieval_rank"), "tool_response": BASE.preview(call.get("response")),
                        "tool_error": call.get("error"), "qos": qos, "cost": cost, "latency": latency,
                        "judge_status": job.get("status"), "judge_verdict": judgment.get("verdict"),
                        "judge_explanation": judgment.get("explanation"),
                        "judge_error": job.get("error"),
                        "candidate_scores": [{
                            "tool": item["tool"],
                            "retrieval_rank": item.get("retrieval_rank"),
                            "retrieval_score": item.get("retrieval_score"),
                            "ucb_score": item.get("ucb_score"),
                            "lqm_score": item.get("lqm_score"),
                            "expected_cost": item.get("expected_cost"),
                            "expected_latency": item.get("expected_latency"),
                        } for item in candidates]}
                history.append(step)
                excluded.append(name)
                remaining = max(0.0, remaining - cost)
                total_cost += cost
                total_steps += 1
            if not final_answer and history:
                synthesis = llm_json(
                    planner_prompt(query, [], history).replace("AVAILABLE_TOOLS:\n[]", "No more calls are allowed. Return action=finish with the best supported answer."),
                    model=args.model, timeout=args.llm_timeout, retries=args.llm_retries)
                final_answer = str(synthesis.get("answer") or "")
            final_judgment = llm_json(final_judge_prompt(query, final_answer, history), model=args.model,
                                      timeout=args.llm_timeout, retries=args.llm_retries)
            passed = (
                final_judgment.get("passed") is True
                and final_judgment.get("verdict") == "pass"
            )
            successes += int(passed)
            record = {"position": args.offset + position - 1, "query_id": row["query_id"],
                      "query": query, "cluster_ids": clusters,
                      "candidate_scope": args.candidate_scope,
                      "candidate_tools": candidate_names if args.candidate_scope == "annotated" else None,
                      "steps": history, "final_answer": final_answer, "final_judgment": final_judgment,
                      "passed": passed, "planner_error": planner_error, "remaining_budget": remaining}
            line = json.dumps(record, ensure_ascii=False) + "\n"
            stream.write(line); stream.flush()
            for cluster_id in clusters:
                folder = args.output_dir / "clusters" / safe_name(cluster_id)
                folder.mkdir(parents=True, exist_ok=True)
                cluster_path = folder / "trajectories.jsonl"
                mode = "a" if cluster_id in initialized_cluster_files else "w"
                with cluster_path.open(mode, encoding="utf-8", newline="\n") as cluster_file:
                    cluster_file.write(line)
                initialized_cluster_files.add(cluster_id)
                stat = cluster_stats[cluster_id]; stat["queries"] += 1; stat["successes"] += int(passed)
                stat["steps"] += len(history); stat["cost"] += sum(x["cost"] for x in history)
            elapsed = time.monotonic() - benchmark_started
            mean_seconds = elapsed / pending_position
            eta_seconds = max(0.0, mean_seconds * (len(pending_rows) - pending_position))
            print(
                "      [{}/{}] query_id={} passed={} steps={} cost={:.6f} "
                "elapsed={:.1f}m eta={:.1f}m".format(
                    position, len(rows), row["query_id"], passed, len(history),
                    sum(x["cost"] for x in history), elapsed / 60.0,
                    eta_seconds / 60.0,
                ),
                flush=True,
            )

    print("[4/5] Writing per-cluster summaries ...", flush=True)
    for cluster_id, stat in cluster_stats.items():
        stat["success_rate"] = stat["successes"] / stat["queries"]
        folder = args.output_dir / "clusters" / safe_name(cluster_id)
        (folder / "summary.json").write_text(json.dumps(stat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latencies = [step["latency"] for line in trajectories_path.read_text(encoding="utf-8").splitlines()
                 for step in json.loads(line)["steps"]]
    sorted_latencies = sorted(latencies)
    p95_latency = (
        sorted_latencies[min(len(sorted_latencies) - 1, int(0.95 * len(sorted_latencies)))]
        if sorted_latencies else None
    )
    summary = {"benchmark": "qostoolbench_agentic_clusters", "run_id": run_id,
               "dataset": str(args.dataset.resolve()), "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
               "qos_profiles": str(args.qos_profiles.resolve()), "queries": len(rows),
               "successes": successes, "success_rate": successes / len(rows) if rows else None,
               "steps": total_steps, "mean_steps": total_steps / len(rows) if rows else None,
               "total_cost": total_cost, "mean_cost": total_cost / len(rows) if rows else None,
               "mean_latency": sum(latencies) / len(latencies) if latencies else None,
               "p95_latency": p95_latency,
               "clusters": len(cluster_stats), "model": args.model, "max_steps": args.max_steps,
               "selection_policy": args.selection_policy, "seed": args.seed,
               "candidate_scope": args.candidate_scope,
               "catalog": str(args.catalog.resolve()) if args.catalog else None,
               "catalog_tools": len(definitions),
               "retrieval_limit": args.retrieval_limit,
               "router_parameters": {"alpha": health.get("alpha"),
                                     "cost_weight": health.get("cost_weight"),
                                     "latency_weight": health.get("latency_weight"),
                                     "lqm_latency_reference": health.get("lqm_latency_reference"),
                                     "lqm_deflation": health.get("lqm_deflation")},
               "request_budget": args.request_budget, "latency_sla": args.latency_sla,
               "qos_server_profiles": qos_status.get("profiles"),
               "qos_server_profiles_file": qos_status.get("profiles_file"),
               "trajectories": str(trajectories_path.resolve())}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "dataset": summary["dataset"], "dataset_sha256": summary["dataset_sha256"],
        "qos_profiles": summary["qos_profiles"], "offset": args.offset, "limit": args.limit,
        "catalog": summary["catalog"], "catalog_tools": summary["catalog_tools"],
        "cluster_filter": args.cluster_id or [],
        "source_query_filter": args.source_query_id,
        "candidate_scope": args.candidate_scope,
        "selection_policy": args.selection_policy,
        "agent_selects_tools": args.selection_policy == "agentic",
        "agent_prompt_version": AGENT_PROMPT_VERSION,
        "agent_prompt_sha256": hashlib.sha256(
            AGENT_SELECTION_INSTRUCTIONS.encode("utf-8")
        ).hexdigest(),
        "toolbandit_role": "retrieval, feasibility and UCB annotation",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[5/5] Result\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
