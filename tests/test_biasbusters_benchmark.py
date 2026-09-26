import importlib.util
from pathlib import Path

import numpy as np

from budget_tool_router.retriever import SemanticToolRetriever, toolbench_documents


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_biasbusters_retrieval_benchmark.py"
SPEC = importlib.util.spec_from_file_location("biasbusters_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeEncoder:
    def encode_many(self, texts):
        result = []
        for text in texts:
            result.append(np.array([1.0, 0.0]) if "weather" in text.lower() else np.array([0.0, 1.0]))
        return result


def api(tool, operation, description):
    return {
        "category_name": "test", "tool_name": tool, "api_name": operation,
        "api_description": description, "required_parameters": [],
        "optional_parameters": [], "method": "GET", "template_response": {},
    }


def test_evaluate_reports_relevant_retrieval():
    weather = api("weather", "current", "weather forecast")
    books = api("books", "search", "book title search")
    rows = [{
        "query": "weather in Moscow", "query_id": 1,
        "api_list": [weather, books, books, books, books],
        "relevant APIs": [["weather", "current"]],
    }]
    retriever = SemanticToolRetriever(FakeEncoder(), toolbench_documents(rows))
    result = MODULE.evaluate(rows, retriever, top_k=1)
    assert result["top1_relevant_rate"] == 1.0
    assert result["hit_rate_at_k"] == 1.0
    assert result["top1_source_position_counts"]["1"] == 1
