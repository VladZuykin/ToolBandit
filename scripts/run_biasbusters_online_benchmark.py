"""Run BiasBusters end to end through ToolBandit and StableToolBenchQoS."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import http.client
import json
import os
import random
from pathlib import Path
import re
import statistics
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid


RESERVED_NAMES = {"from", "class", "return", "false", "true", "id", "and"}


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 180.0,
    retries: int = 0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    for retry_index in range(retries + 1):
        request = Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read()
                return {} if not body else json.loads(body.decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code not in {408, 429, 500, 502, 503, 504} or retry_index >= retries:
                raise RuntimeError("HTTP {} {}: {}".format(error.code, url, detail)) from error
        except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
            if retry_index >= retries:
                reason = getattr(error, "reason", error)
                raise RuntimeError("cannot reach {}: {}".format(url, reason)) from error
        delay = min(2 ** retry_index, 8)
        print(
            "      Transient HTTP failure for {}; retry {}/{} in {}s ...".format(
                url, retry_index + 1, retries, delay
            ),
            flush=True,
        )
        time.sleep(delay)
    raise RuntimeError("request failed without a response: {}".format(url))


def standardize(value: str) -> str:
    result = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]", "_", value or "")
    result = re.sub(r"_+", "_", result).strip("_").lower()
    if result and result[0].isdigit():
        result = "get_" + result
    return result


def api_id(api: dict[str, Any]) -> str:
    if api.get("_api_id"):
        return str(api["_api_id"])
    category = re.sub(
        r"_+", "_", str(api.get("category_name") or "").replace(" ", "_").replace(",", "_").replace("/", "_")
    )
    name = standardize(str(api.get("api_name") or ""))
    if name in RESERVED_NAMES:
        name = "is_" + name
    return "{}/{}/{}".format(category, standardize(str(api["tool_name"])), name)


def tool_name(api: dict[str, Any]) -> str:
    return "{}::{}".format(api["tool_name"], api["api_name"])


def parameter_schema(api: dict[str, Any]) -> dict[str, Any]:
    type_map = {
        "STRING": "string", "NUMBER": "number", "INTEGER": "integer",
        "BOOLEAN": "boolean", "ARRAY": "array", "OBJECT": "object",
    }
    properties: dict[str, Any] = {
        "query": {
            "type": "string",
            "description": "Original user request supplied to the virtual tool simulator.",
        }
    }
    required = []
    for field_name, is_required in (("required_parameters", True), ("optional_parameters", False)):
        for parameter in api.get(field_name, []):
            name = str(parameter.get("name") or "")
            if not name:
                continue
            definition: dict[str, Any] = {
                "type": type_map.get(str(parameter.get("type") or "").upper(), "string")
            }
            if parameter.get("description"):
                definition["description"] = str(parameter["description"])
            if parameter.get("default") is not None:
                definition["default"] = parameter["default"]
            properties[name] = definition
            if is_required:
                required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def load_profiles(path: Path, scenario: str) -> dict[str, dict[str, float]]:
    profiles = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            values = row["simulation"]["profiles"][scenario]
            profiles[str(row["api_id"])] = {
                "cost": float(values["cost_per_call_units"]),
                "latency": float(values["expected_latency_ms"]) / 1000.0,
                "success_rate": float(values["success_rate"]),
            }
    return profiles


def unique_apis(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        for api in row["api_list"]:
            result.setdefault(tool_name(api), api)
    return result


def load_benchmark_rows(dataset_path: Path, clusters_path: Path | None) -> list[dict[str, Any]]:
    """Load either expanded BiasBusters rows or the original clustered queries."""
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not payload or "queries" not in payload[0]:
        return payload
    if clusters_path is None:
        raise ValueError("--clusters is required when --dataset points to cluster_queries.json")
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for cluster_queries in payload:
        cluster_id = int(cluster_queries["cluster_id"])
        if cluster_id < 1 or cluster_id > len(clusters):
            raise ValueError("cluster_id {} is absent from {}".format(cluster_id, clusters_path))
        api_list = []
        for api in clusters[cluster_id - 1]:
            normalized = dict(api)
            normalized["category_name"] = normalized.pop("category")
            normalized["tool_name"] = normalized.pop("tool")
            api_list.append(normalized)
        for query_index, query in enumerate(cluster_queries["queries"], start=1):
            rows.append({
                "query_id": "{}-{}".format(cluster_id, query_index),
                "cluster_id": cluster_id,
                "query": query,
                "api_list": api_list,
            })
    return rows


def catalog_definitions(
    apis: dict[str, dict[str, Any]], profiles: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    definitions = []
    missing = []
    for name, api in sorted(apis.items()):
        profile = profiles.get(api_id(api))
        if profile is None:
            missing.append(api_id(api))
            continue
        output_schema = api.get("template_response")
        definitions.append({
            "tool_name": name,
            "description": "{}\nTool: {}\nOperation: {}".format(
                api.get("api_description", ""), api["tool_name"], api["api_name"]
            ),
            "input_schema": parameter_schema(api),
            "output_schema": output_schema if isinstance(output_schema, dict) else {},
            "cost": profile["cost"],
            "latency": profile["latency"],
            "metadata": {
                "api_id": api_id(api),
                "category_name": api.get("category_name", ""),
                "tool_name": api["tool_name"],
                "api_name": api["api_name"],
                "api_description": api.get("api_description", ""),
                "required_parameters": api.get("required_parameters", []),
                "optional_parameters": api.get("optional_parameters", []),
                "template_response": api.get("template_response", {}),
            },
            "enabled": True,
        })
    if missing:
        raise ValueError("QoS profiles missing for {} APIs: {}".format(len(missing), missing[:3]))
    return definitions


def default_tool_arguments(query: str, metadata: dict[str, Any]) -> dict[str, Any]:
    arguments: dict[str, Any] = {"query": query}
    for parameter in metadata.get("required_parameters", []):
        name = str(parameter.get("name") or "")
        if name and parameter.get("default") is not None:
            arguments[name] = parameter["default"]
    return arguments


def _json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("argument generator did not return a JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("generated tool arguments are not an object")
    return value


def generate_tool_arguments(
    query: str, metadata: dict[str, Any], *, model: str, timeout: float,
    retries: int = 4,
) -> dict[str, Any]:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for LLM argument generation")
    parameters = {
        "required_parameters": metadata.get("required_parameters", []),
        "optional_parameters": metadata.get("optional_parameters", []),
    }
    prompt = (
        "Generate arguments for the selected API from the user request. "
        "Return only one JSON object. Use the API parameter names exactly. "
        "Do not invent a value when it cannot be inferred.\n"
        "User request: " + query + "\n"
        "API: " + str(metadata.get("tool_name")) + " / " + str(metadata.get("api_name")) + "\n"
        "Parameters: " + json.dumps(parameters, ensure_ascii=False)
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You convert user requests into valid API arguments."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    result = None
    for retry_index in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            # Retry throttling and transient provider failures, but fail fast on
            # invalid credentials or malformed requests.
            if error.code not in {408, 429, 500, 502, 503, 504} or retry_index >= retries:
                raise RuntimeError(
                    "DeepSeek argument generation failed: HTTP {}: {}".format(error.code, detail)
                ) from error
        except (
            URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
        ) as error:
            if retry_index >= retries:
                raise RuntimeError(
                    "DeepSeek argument generation failed after {} attempts: {}".format(
                        retries + 1, error
                    )
                ) from error
        delay = min(2 ** retry_index, 16)
        print(
            "      Argument generator temporarily unavailable; retry {}/{} in {}s ...".format(
                retry_index + 1, retries, delay
            ),
            flush=True,
        )
        time.sleep(delay)
    if result is None:
        raise RuntimeError("DeepSeek argument generation returned no result")
    return _json_object(result["choices"][0]["message"]["content"])


def preview(value: Any, limit: int = 4000) -> Any:
    encoded = json.dumps(value, ensure_ascii=False)
    if len(encoded) <= limit:
        return value
    return encoded[:limit] + "...[truncated]"


def profile_for_candidate(
    candidate: dict[str, Any], profiles: dict[str, dict[str, float]]
) -> dict[str, float]:
    metadata = candidate.get("metadata") or {}
    identity = str(metadata.get("api_id") or api_id({
        "category_name": metadata.get("category_name", ""),
        "tool_name": metadata.get("tool_name", ""),
        "api_name": metadata.get("api_name", ""),
    }))
    return profiles[identity]


def select_candidate(
    candidates: list[dict[str, Any]], *, policy: str,
    profiles: dict[str, dict[str, float]], rng: random.Random,
    cost_weight: float, latency_weight: float,
    cost_scale: float, latency_scale: float,
) -> dict[str, Any]:
    if policy in {"ucb", "ucb-top1"}:
        return max(candidates, key=lambda item: (float(item["ucb_score"]), -int(item["retrieval_rank"])))
    if policy == "lqm-context-route":
        return max(candidates, key=lambda item: (float(item["lqm_score"]), -int(item["retrieval_rank"])))
    if policy == "semantic":
        return candidates[0]
    if policy == "random":
        return rng.choice(candidates)
    if policy == "cheapest":
        return min(candidates, key=lambda item: (profile_for_candidate(item, profiles)["cost"], item["retrieval_rank"]))
    if policy == "fastest":
        return min(candidates, key=lambda item: (profile_for_candidate(item, profiles)["latency"], item["retrieval_rank"]))
    if policy == "highest-pass-rate":
        return max(candidates, key=lambda item: (profile_for_candidate(item, profiles)["success_rate"], -item["retrieval_rank"]))
    if policy == "oracle-utility":
        def utility(item: dict[str, Any]) -> float:
            profile = profile_for_candidate(item, profiles)
            return (
                profile["success_rate"]
                - cost_weight * profile["cost"] / cost_scale
                - latency_weight * profile["latency"] / latency_scale
            )
        return max(candidates, key=lambda item: (utility(item), -item["retrieval_rank"]))
    raise ValueError("unknown selection policy: {}".format(policy))


def expected_contract(metadata: dict[str, Any]) -> str:
    schema = metadata.get("template_response") or {}
    return "Return a useful answer for the user request matching this expected response structure: " + json.dumps(
        schema, ensure_ascii=False
    )


def wait_for_judge(
    base_url: str, evaluation_id: str, timeout: float, poll_interval: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        job = request_json(
            base_url.rstrip("/") + "/v1/evaluations/" + evaluation_id,
            retries=4,
        )
        if job.get("status") != "pending":
            return job
        if time.monotonic() >= deadline:
            raise RuntimeError("Judge timeout for {}".format(evaluation_id))
        time.sleep(poll_interval)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--clusters", type=Path,
        help="duplicate_api_clusters.json; required for the original cluster_queries.json format",
    )
    parser.add_argument("--qos-profiles", type=Path, required=True)
    parser.add_argument("--toolbandit-url", default="http://127.0.0.1:8080")
    parser.add_argument("--toolbench-url", default="http://127.0.0.1:8082/virtual")
    parser.add_argument("--scenario", choices=("normal", "degraded", "outage"), default="normal")
    parser.add_argument("--output", type=Path, default=Path("data/biasbusters_online_results.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("data/biasbusters_online_summary.json"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--samples-per-cluster", type=int,
        help="Deterministically sample this many queries from every cluster before offset/limit.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retrieval-limit", type=int, default=5)
    parser.add_argument("--request-budget", type=float, default=0.05)
    parser.add_argument("--latency-sla", type=float, default=300.0)
    parser.add_argument("--judge-timeout", type=float, default=180.0)
    parser.add_argument("--judge-poll-interval", type=float, default=1.0)
    parser.add_argument("--search-retries", type=int, default=4)
    parser.add_argument(
        "--selection-policy",
        choices=("ucb", "ucb-top1", "semantic", "random", "cheapest", "fastest",
                 "highest-pass-rate", "oracle-utility", "lqm-context-route"),
        default="ucb",
    )
    parser.add_argument("--oracle-cost-weight", type=float, default=0.5)
    parser.add_argument("--oracle-latency-weight", type=float, default=0.5)
    parser.add_argument("--argument-generator", choices=("deepseek", "defaults"), default="deepseek")
    parser.add_argument("--argument-model", default="deepseek-chat")
    parser.add_argument("--argument-timeout", type=float, default=60.0)
    parser.add_argument("--argument-retries", type=int, default=4)
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument(
        "--reset-qos", action="store_true",
        help="Reset StableToolBenchQoS deterministic call counters before the run.",
    )
    parser.add_argument(
        "--reset-catalog",
        action="store_true",
        help="Delete all registered tools before importing the 50 benchmark tools.",
    )
    args = parser.parse_args()
    if args.offset < 0 or args.limit is not None and args.limit < 0:
        parser.error("offset and limit must be non-negative")

    rows = load_benchmark_rows(args.dataset, args.clusters)
    if args.samples_per_cluster is not None:
        if args.samples_per_cluster < 1:
            parser.error("samples-per-cluster must be positive")
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            if "cluster_id" not in row:
                parser.error("samples-per-cluster requires rows containing cluster_id")
            grouped.setdefault(int(row["cluster_id"]), []).append(row)
        rng = random.Random(args.seed)
        sampled = []
        for cluster_id in sorted(grouped):
            cluster_rows = grouped[cluster_id]
            if args.samples_per_cluster > len(cluster_rows):
                parser.error(
                    "cluster {} contains only {} queries".format(cluster_id, len(cluster_rows))
                )
            sampled.extend(rng.sample(cluster_rows, args.samples_per_cluster))
        rng.shuffle(sampled)
        rows = sampled
    selected_rows = rows[args.offset:]
    if args.limit is not None:
        selected_rows = selected_rows[:args.limit]
    profiles = load_profiles(args.qos_profiles, args.scenario)
    apis = unique_apis(rows)
    definitions = catalog_definitions(apis, profiles)
    toolbandit = args.toolbandit_url.rstrip("/")

    print("[1/5] Checking services ...", flush=True)
    health = request_json(toolbandit + "/health")
    if health.get("status") != "ok" or not health.get("judge") or not health.get("registry"):
        raise RuntimeError("ToolBandit is not ready: {}".format(health))
    with urlopen(args.toolbench_url.rsplit("/", 1)[0] + "/docs", timeout=10):
        pass
    print("      ToolBandit and StableToolBenchQoS are reachable")
    if args.reset_qos:
        reset_result = request_json(
            args.toolbench_url.rsplit("/", 1)[0] + "/qos/reset", method="POST"
        )
        print("      QoS sequences reset: {}".format(reset_result), flush=True)

    if not args.skip_import:
        if args.reset_catalog:
            existing = request_json(toolbandit + "/v1/tools?limit=1000").get("items", [])
            print("[2/5] Clearing {} existing tools ...".format(len(existing)), flush=True)
            for item in existing:
                request_json(
                    toolbandit + "/v1/tools/by-name?" + urlencode({"tool_name": item["tool_name"]}),
                    method="DELETE",
                    timeout=600,
                )
        print("[2/5] Importing {} BiasBusters tools ...".format(len(definitions)), flush=True)
        imported = request_json(
            toolbandit + "/v1/tools/import",
            method="POST",
            payload={"tools": definitions, "replace_existing": True},
            timeout=600,
        )
        print("      {}".format(imported))
    else:
        print("[2/5] Tool import skipped")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    selection_rng = random.Random(args.seed)
    attempts_total = successes = no_candidate = judge_failures = 0
    cluster_metrics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "queries": 0, "successes": 0, "attempts": 0,
            "total_cost": 0.0, "latencies": [],
        }
    )
    total_cost = 0.0
    latencies: list[float] = []
    selected_tools: Counter[str] = Counter()
    print("[3/5] Running {} queries from offset {} ...".format(len(selected_rows), args.offset), flush=True)
    benchmark_started = time.monotonic()
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        for position, row in enumerate(selected_rows, start=1):
            query = str(row["query"])
            cluster_key = str(row.get("cluster_id", "expanded"))
            cluster_metric = cluster_metrics[cluster_key]
            cluster_metric["queries"] += 1
            relevant_tools = [tool_name(api) for api in row["api_list"]]
            context = query
            remaining_budget = args.request_budget
            excluded: list[str] = []
            attempts = []
            passed = False
            for attempt_index in range(1, args.max_attempts + 1):
                search = request_json(
                    toolbandit + "/v1/tools/search",
                    method="POST",
                    payload={
                        "query": context,
                        "remaining_budget": remaining_budget,
                        "latency_sla": args.latency_sla,
                        "retrieval_limit": args.retrieval_limit,
                        "result_limit": args.retrieval_limit,
                        "retrieval_threshold": -1.0,
                        "excluded_tools": excluded,
                        "candidate_tools": relevant_tools,
                    },
                    retries=args.search_retries,
                )
                candidates = search.get("tools", [])
                if not candidates:
                    no_candidate += 1
                    break
                choice = select_candidate(
                    candidates, policy=args.selection_policy, profiles=profiles,
                    rng=selection_rng,
                    cost_weight=args.oracle_cost_weight,
                    latency_weight=args.oracle_latency_weight,
                    cost_scale=float(health.get("cost_scale", 0.012)),
                    latency_scale=float(health.get("latency_scale", 3.0)),
                )
                name = str(choice["tool"])
                metadata = dict(choice.get("metadata") or {})
                if args.argument_generator == "deepseek":
                    arguments = generate_tool_arguments(
                        query, metadata, model=args.argument_model,
                        timeout=args.argument_timeout, retries=args.argument_retries,
                    )
                else:
                    arguments = default_tool_arguments(query, metadata)
                call_payload = {
                    "category": metadata["category_name"],
                    "tool_name": metadata["tool_name"],
                    "api_name": metadata["api_name"],
                    "tool_input": arguments,
                    "strip": "",
                    "toolbench_key": "",
                }
                call = request_json(
                    args.toolbench_url, method="POST", payload=call_payload,
                    timeout=180, retries=4,
                )
                qos = dict(call.get("qos") or {})
                observed_cost = float(qos.get("cost_units") or 0.0)
                observed_latency = float(qos.get("latency_ms") or 0.0) / 1000.0
                evaluation = request_json(
                    toolbandit + "/v1/evaluations",
                    method="POST",
                    retries=4,
                    payload={
                        "interaction_id": "bb-{}-{}-{}-{}".format(
                            run_id, row["query_id"], args.offset + position, attempt_index
                        ),
                        "request_id": search["request_id"],
                        "tool_name": name,
                        "tool_intent": query,
                        "expected_contract": expected_contract(metadata),
                        "tool_output": call,
                        "observed_cost": observed_cost,
                        "observed_latency": observed_latency,
                    },
                )
                job = wait_for_judge(
                    toolbandit, str(evaluation["evaluation_id"]),
                    args.judge_timeout, args.judge_poll_interval,
                )
                judgment = dict(job.get("judgment") or {})
                learned = bool(job.get("learned"))
                passed = job.get("status") == "completed" and bool(judgment.get("passed"))
                if job.get("status") != "completed":
                    judge_failures += 1
                attempt = {
                    "attempt": attempt_index,
                    "tool": name,
                    "retrieval_rank": choice.get("retrieval_rank"),
                    "retrieval_score": choice.get("retrieval_score"),
                    "ucb_score": choice.get("ucb_score"),
                    "qos": qos,
                    "tool_input": arguments,
                    "tool_response": preview(call.get("response")),
                    "tool_error": call.get("error"),
                    "observed_cost": observed_cost,
                    "observed_latency": observed_latency,
                    "judge_status": job.get("status"),
                    "judge_error": job.get("error"),
                    "judge_verdict": judgment.get("verdict"),
                    "judge_confidence": judgment.get("confidence"),
                    "judge_reason_code": judgment.get("reason_code"),
                    "judge_explanation": judgment.get("explanation"),
                    "judge_missing_requirements": judgment.get("missing_requirements"),
                    "learned": learned,
                    "passed": passed,
                }
                attempts.append(attempt)
                attempts_total += 1
                cluster_metric["attempts"] += 1
                total_cost += observed_cost
                cluster_metric["total_cost"] += observed_cost
                latencies.append(observed_latency)
                cluster_metric["latencies"].append(observed_latency)
                selected_tools[name] += 1
                remaining_budget = max(0.0, remaining_budget - observed_cost)
                excluded.append(name)
                if passed:
                    successes += 1
                    cluster_metric["successes"] += 1
                    break
                context = (
                    query + "\nPREVIOUS_ATTEMPT:\nTool: " + name
                    + "\nQoS succeeded: " + str(qos.get("succeeded"))
                    + "\nError: " + str(call.get("error", ""))
                    + "\nPartial output: " + json.dumps(call.get("response"), ensure_ascii=False)[:2000]
                )
            record = {
                "position": args.offset + position - 1,
                "query_id": row["query_id"],
                "cluster_id": row.get("cluster_id"),
                "query": query,
                "passed": passed,
                "attempts": attempts,
                "remaining_budget": remaining_budget,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            elapsed = time.monotonic() - benchmark_started
            eta = elapsed / position * (len(selected_rows) - position) if position else 0.0
            print(
                "      [{}/{}] query_id={} passed={} attempts={} cost={:.6f} elapsed={:.1f}m eta={:.1f}m".format(
                    position, len(selected_rows), row["query_id"], passed,
                    len(attempts), sum(item["observed_cost"] for item in attempts),
                    elapsed / 60.0, eta / 60.0,
                ),
                flush=True,
            )

    print("[4/5] Calculating metrics ...", flush=True)
    count = len(selected_rows)
    per_cluster = {}
    for cluster_id, metric in sorted(cluster_metrics.items(), key=lambda item: item[0]):
        cluster_count = int(metric["queries"])
        cluster_latencies = list(metric["latencies"])
        per_cluster[cluster_id] = {
            "queries": cluster_count,
            "successes": metric["successes"],
            "success_rate": metric["successes"] / cluster_count if cluster_count else None,
            "attempts": metric["attempts"],
            "mean_attempts": metric["attempts"] / cluster_count if cluster_count else None,
            "total_cost": metric["total_cost"],
            "mean_cost": metric["total_cost"] / cluster_count if cluster_count else None,
            "mean_latency": statistics.mean(cluster_latencies) if cluster_latencies else None,
            "p95_latency": percentile(cluster_latencies, 0.95),
        }
    summary = {
        "benchmark": "biasbusters_toolbandit_online",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "clusters": str(args.clusters.resolve()) if args.clusters else None,
        "clusters_sha256": hashlib.sha256(args.clusters.read_bytes()).hexdigest() if args.clusters else None,
        "qos_profiles": str(args.qos_profiles.resolve()),
        "scenario": args.scenario,
        "qos_reset": args.reset_qos,
        "run_id": run_id,
        "selection_policy": args.selection_policy,
        "router_parameters": {
            "alpha": health.get("alpha"),
            "cost_weight": health.get("cost_weight"),
            "latency_weight": health.get("latency_weight"),
            "cost_scale": health.get("cost_scale"),
            "latency_scale": health.get("latency_scale"),
        },
        "offset": args.offset,
        "samples_per_cluster": args.samples_per_cluster,
        "seed": args.seed,
        "queries": count,
        "successes": successes,
        "success_rate": successes / count if count else None,
        "attempts": attempts_total,
        "mean_attempts": attempts_total / count if count else None,
        "total_cost": total_cost,
        "mean_cost": total_cost / count if count else None,
        "mean_latency": statistics.mean(latencies) if latencies else None,
        "p95_latency": percentile(latencies, 0.95),
        "queries_without_candidate": no_candidate,
        "judge_failures": judge_failures,
        "most_selected_tools": selected_tools.most_common(20),
        "per_cluster": per_cluster,
        "request_budget": args.request_budget,
        "latency_sla": args.latency_sla,
        "max_attempts": args.max_attempts,
        "retrieval_limit": args.retrieval_limit,
        "argument_generator": args.argument_generator,
        "argument_model": args.argument_model if args.argument_generator == "deepseek" else None,
        "details": str(args.output.resolve()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[5/5] Result\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
