"""Benchmark ToolBandit's semantic retrieval on BiasBusters ToolBench queries."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from budget_tool_router.ada_embeddings import Ada002Encoder  # noqa: E402
from budget_tool_router.retriever import (  # noqa: E402
    SemanticToolRetriever,
    toolbench_documents,
)


def tool_id(tool_name: str, api_name: str) -> str:
    return f"{tool_name}::{api_name}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(rows: list[dict], retriever: SemanticToolRetriever, *, top_k: int) -> dict:
    query_vectors: dict[str, np.ndarray] = {}
    unique_queries = list(dict.fromkeys(str(row["query"]) for row in rows))
    vectors = retriever.encoder.encode_many(unique_queries)
    query_vectors.update(zip(unique_queries, vectors))

    top1_hits = 0
    hit_at_k = 0
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    selected_tools: Counter[str] = Counter()
    source_positions: Counter[int] = Counter()

    for row in rows:
        relevant = {tool_id(str(item[0]), str(item[1])) for item in row["relevant APIs"]}
        hits = retriever.search_vector(query_vectors[str(row["query"])], limit=len(retriever.documents))
        ranked = [hit.tool_name for hit in hits]
        first_rank = next((index for index, name in enumerate(ranked, 1) if name in relevant), None)
        if first_rank is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / first_rank)
        top = ranked[:top_k]
        overlap = relevant.intersection(top)
        top1_hits += int(bool(ranked) and ranked[0] in relevant)
        hit_at_k += int(bool(overlap))
        recalls.append(len(overlap) / len(relevant))
        if ranked:
            selected_tools[ranked[0]] += 1
            candidates = [tool_id(str(api["tool_name"]), str(api["api_name"])) for api in row["api_list"]]
            if ranked[0] in candidates:
                source_positions[candidates.index(ranked[0]) + 1] += 1

    count = len(rows)
    return {
        "records": count,
        "unique_query_texts": len(unique_queries),
        "tools": len(retriever.documents),
        "top_k": top_k,
        "top1_relevant_rate": top1_hits / count,
        "hit_rate_at_k": hit_at_k / count,
        "mean_recall_at_k": float(np.mean(recalls)),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "top1_tool_counts": dict(selected_tools.most_common()),
        "top1_source_position_counts": {
            str(position): source_positions[position] for position in range(1, 6)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "biasbusters_retrieval_results.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    dataset = args.dataset.expanduser().resolve()
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("dataset is empty")

    print(f"[1/4] Dataset: {dataset} ({len(rows)} rows)")
    documents = toolbench_documents(rows)
    print(f"[2/4] Unique ToolBench endpoints: {len(documents)}")
    encoder = Ada002Encoder(os.environ.get("DATABASE_URL"), output_dimension=1536)
    print("[3/4] Embedding tools and unique queries (cache is reused) ...")
    retriever = SemanticToolRetriever(encoder, documents)
    metrics = evaluate(rows, retriever, top_k=args.top_k)
    result = {
        "benchmark": "biasbusters_toolbench_semantic_retrieval",
        "dataset": str(dataset),
        "dataset_sha256": file_sha256(dataset),
        "embedding_model": encoder.model,
        "embedding_dimension": encoder.output_dimension,
        "scope": "retrieval_only_no_tool_execution_no_linucb_update",
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[4/4] Result")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
