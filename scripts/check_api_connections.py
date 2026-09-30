"""Make one real request to each paid provider and report safe diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from budget_tool_router.ada_embeddings import Ada002Encoder, MemoryEmbeddingCache
from budget_tool_router import DeepSeekJudge


def main() -> int:
    failures = 0
    print("[1/2] OpenAI embeddings: checking OPENAI_API_KEY ...", flush=True)
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_KEY"):
        print("      FAIL: OPENAI_API_KEY is not set")
        failures += 1
    else:
        try:
            encoder = Ada002Encoder(cache=MemoryEmbeddingCache())
            started = time.perf_counter()
            vector = encoder.encode_many(["ToolBandit API connection test"])[0]
            elapsed = time.perf_counter() - started
            print("      OK: model={}, dimension={}, latency={:.3f}s".format(
                encoder.model, len(vector), elapsed))
        except Exception as error:
            print("      FAIL: {}: {}".format(type(error).__name__, error))
            failures += 1

    print("[2/2] DeepSeek judge: checking DEEPSEEK_API_KEY ...", flush=True)
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("      FAIL: DEEPSEEK_API_KEY is not set")
        failures += 1
    else:
        try:
            started = time.perf_counter()
            result = DeepSeekJudge().evaluate(
                query="Read a health-check value",
                expected_contract="Return status equal to ok.",
                tool_name="connection_test",
                tool_output={"status": "ok"},
            )
            elapsed = time.perf_counter() - started
            print("      OK: model={}, verdict={}, confidence={:.2f}, latency={:.3f}s, tokens={}".format(
                result.model, result.verdict, result.confidence, elapsed, result.total_tokens))
        except Exception as error:
            print("      FAIL: {}: {}".format(type(error).__name__, error))
            failures += 1

    print("\nResult: {}".format("all connections work" if not failures else "{} check(s) failed".format(failures)))
    print("API keys were not printed or persisted.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
