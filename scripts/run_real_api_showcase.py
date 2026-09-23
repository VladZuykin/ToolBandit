"""Verbose, manual end-to-end run against ten public JSON APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from budget_tool_router.ada_embeddings import Ada002Encoder
from budget_tool_router import (
    BudgetAwareToolRouter, DeepSeekJudge, SemanticToolRetriever,
    ToolDocument, ToolRoutingService, ToolSpec,
)
from budget_tool_router.adapters import public_api_examples

ROOT = PROJECT_ROOT


def preview(value, limit=500):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + " ..."


def wait_for(service, evaluation_id):
    while True:
        job = service.get_evaluation(evaluation_id)
        if job["status"] != "pending":
            return job
        time.sleep(0.1)


def find_row(search, tool_name):
    return next((row for row in search["tools"] if row["tool"] == tool_name), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="number of real APIs to call")
    parser.add_argument("--skip-judge", action="store_true", help="do not call DeepSeek")
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()

    examples = public_api_examples(timeout=args.timeout)[:max(0, min(args.limit, 10))]
    print("[1/6] CATALOG: prepared {} real read-only API adapters".format(len(examples)), flush=True)
    for number, example in enumerate(examples, 1):
        print("      {:2d}. {}".format(number, example.adapter.tool_name))

    print("\n[2/6] EMBEDDINGS: embedding tool descriptions with text-embedding-ada-002 ...", flush=True)
    encoder = Ada002Encoder(ROOT / "data" / "ada002_showcase_cache.sqlite3", output_dimension=1536)
    documents = []
    for example in examples:
        adapter = example.adapter
        documents.append(ToolDocument(
            adapter.tool_name,
            json.dumps({"description": adapter.description,
                        "input_schema": adapter.input_schema,
                        "output_schema": adapter.output_schema}, ensure_ascii=False),
            adapter.catalog_entry(),
        ))
    retriever = SemanticToolRetriever(encoder, documents)
    print("      OK: {} tool vectors, dimension=1536".format(len(documents)))

    specs = [ToolSpec(e.adapter.tool_name, 0.0, None, fixed_cost=0.0,
                      latency_range=args.timeout, prior_count=1) for e in examples]
    router = BudgetAwareToolRouter(
        specs, context_dimension=1536, diagonal_covariance=True,
        cost_weight=0.35, latency_weight=0.65,
        cost_scale=0.01, latency_scale=args.timeout,
    )
    judge = None if args.skip_judge else DeepSeekJudge()
    service = ToolRoutingService(retriever, router, judge)
    print("\n[3/6] ROUTER: diagonal LinUCB created; cost_weight=.35, latency_weight=.65")
    print("      Cold-start latency estimate is the enforceable {:.1f}s timeout.".format(args.timeout))

    successes = 0
    judged_passes = 0
    for index, example in enumerate(examples, 1):
        adapter = example.adapter
        print("\n" + "=" * 78)
        print("API {}/{}: {}".format(index, len(examples), adapter.tool_name))
        print("QUERY: {}".format(example.query))
        estimates = {}
        for candidate in examples:
            estimate = candidate.adapter.estimate(candidate.arguments)
            estimates[candidate.adapter.tool_name] = {
                "estimated_cost": estimate.estimated_cost,
                "estimated_latency": estimate.estimated_latency,
            }

        print("[4/6] RETRIEVE + SCORE: Ada cosine retrieval, then LinUCB reranking", flush=True)
        search = service.search({
            "query": example.query, "remaining_budget": 1.0,
            "latency_sla": args.timeout + 0.01, "retrieval_threshold": -1.0,
            "retrieval_limit": len(examples), "result_limit": len(examples),
            "runtime_estimates": estimates,
        })
        for row in sorted(search["tools"], key=lambda x: x["retrieval_rank"])[:3]:
            print("      retrieval #{:<2} {:<31} cosine={:.4f}  UCB-score={:.4f}".format(
                row["retrieval_rank"], row["tool"], row["retrieval_score"], row["ucb_score"]))
        row = find_row(search, adapter.tool_name)
        if row:
            print("      EXPECTED TOOL: retrieval_rank={}, pass_ucb={:.4f}, ucb_score={:.4f}".format(
                row["retrieval_rank"], row["pass_ucb"], row["ucb_score"]))
        print("      First feasible retrieval result: {}".format(
            search["tools"][0]["tool"] if search["tools"] else None))

        print("[5/6] REAL HTTP CALL: invoking expected semantic tool with {}".format(example.arguments), flush=True)
        execution = adapter.invoke(example.arguments)
        ok = execution.error_code is None and execution.status_code is not None and execution.status_code < 400
        successes += int(ok)
        print("      status={}, error={}, observed_cost={:.6f}, observed_latency={:.3f}s".format(
            execution.status_code, execution.error_code, execution.observed_cost, execution.observed_latency))
        print("      output: {}".format(preview(execution.output if ok else execution.error_message)))

        if judge is not None:
            print("[6/6] ASYNC JUDGE: submitted to DeepSeek; waiting for delayed reward ...", flush=True)
            submitted = service.submit_evaluation({
                "request_id": search["request_id"], "tool_name": adapter.tool_name,
                "tool_intent": example.intent, "expected_contract": example.expected_contract,
                "tool_output": execution.output if ok else {
                    "error_code": execution.error_code, "error_message": execution.error_message},
                "observed_cost": execution.observed_cost,
                "observed_latency": execution.observed_latency,
            })
            job = wait_for(service, submitted["evaluation_id"])
            if job["status"] == "completed":
                judgment = job["judgment"]
                judged_passes += int(judgment["passed"])
                print("      verdict={}, confidence={:.2f}, reward={}, learned={}".format(
                    judgment["verdict"], judgment["confidence"], int(judgment["passed"]), job["learned"]))
                print("      reason: {}".format(judgment["explanation"]))
            else:
                print("      JUDGE FAILED: {}".format(job.get("error")))
        else:
            print("[6/6] ASYNC JUDGE: skipped (--skip-judge); LinUCB is not updated")

    print("\n" + "=" * 78)
    print("SUMMARY: HTTP success {}/{}".format(successes, len(examples)))
    if judge is not None:
        print("         Judge pass {}/{}".format(judged_passes, len(examples)))
    print("Scores shown above are real router scores; API cost is 0 for these public examples.")


if __name__ == "__main__":
    main()
