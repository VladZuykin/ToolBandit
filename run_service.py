"""Start ToolBandit with its persistent API-managed tool registry."""

from __future__ import annotations

import os

from budget_tool_router.ada_embeddings import Ada002Encoder
from budget_tool_router import (
    BudgetAwareToolRouter, DeepSeekJudge, SemanticToolRetriever,
    ToolRoutingService,
)
from budget_tool_router.registry import ToolRegistry, definition_assets
from budget_tool_router.service import serve

def covariance_mode() -> tuple[str, bool]:
    mode = os.getenv("TOOLBANDIT_COVARIANCE", "diagonal").strip().lower()
    if mode not in {"diagonal", "full"}:
        raise SystemExit("TOOLBANDIT_COVARIANCE must be 'diagonal' or 'full'")
    return mode, mode == "diagonal"


def main() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DEEPSEEK_API_KEY is required: ToolBandit needs Judge feedback to train LinUCB"
        )
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required (PostgreSQL)")
    registry = ToolRegistry(database_url)
    active_definitions = registry.list(limit=100_000, enabled=True)
    assets = [definition_assets(item) for item in active_definitions]
    documents = [item[0] for item in assets]
    specs = [item[1] for item in assets]

    encoder = Ada002Encoder(database_url, output_dimension=1536)
    retriever = SemanticToolRetriever(encoder, documents)
    covariance, diagonal_covariance = covariance_mode()
    router = BudgetAwareToolRouter(
        specs, context_dimension=1536,
        alpha=float(os.getenv("TOOLBANDIT_ALPHA", "0.35")), regularization=1.0,
        diagonal_covariance=diagonal_covariance,
        cost_weight=float(os.getenv("TOOL_COST_WEIGHT", "0.5")),
        latency_weight=float(os.getenv("TOOL_LATENCY_WEIGHT", "0.5")),
        cost_scale=float(os.getenv("TOOL_COST_SCALE", "0.012")),
        latency_scale=float(os.getenv("TOOL_LATENCY_SCALE", "3.0")),
    )
    judge = DeepSeekJudge()
    service = ToolRoutingService(retriever, router, judge, registry=registry)
    print(f"Tool registry: {registry.location} ({registry.count()} total, {len(specs)} enabled)")
    print(f"LinUCB covariance: {covariance}")
    serve(service, host=os.getenv("TOOLBANDIT_HOST", "127.0.0.1"),
          port=int(os.getenv("TOOLBANDIT_PORT", "8080")))


if __name__ == "__main__":
    main()
