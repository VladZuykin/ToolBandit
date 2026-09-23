import tempfile
from pathlib import Path

import numpy as np

from budget_tool_router import BudgetAwareToolRouter, SemanticToolRetriever, ToolRoutingService
from budget_tool_router.registry import ToolRegistry


class FakeEncoder:
    def encode_many(self, texts):
        return [np.array([1.0, 0.0, 0.0]) for _ in texts]


def definition(name="weather", description="Current weather"):
    return {
        "tool_name": name, "description": description,
        "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
        "cost": 0.0, "latency": 0.5, "metadata": {},
    }


def test_registry_crud_and_live_index_preserve_learning_on_update():
    with tempfile.TemporaryDirectory() as directory:
        registry = ToolRegistry(Path(directory) / "tools.sqlite3")
        registry.put(definition(), create_only=True)
        retriever = SemanticToolRetriever(FakeEncoder(), [])
        router = BudgetAwareToolRouter([], context_dimension=3, diagonal_covariance=True)
        service = ToolRoutingService(retriever, router, registry=registry)
        assert service.refresh_registry() == 1
        assert router.tool_names == ["weather"]

        router.observe("weather", np.array([1.0, 0.0, 0.0]), passed=True,
                       observed_cost=0.0, observed_latency=0.4)
        updated = definition(description="Updated weather description")
        registry.put(updated)
        service.refresh_registry()
        assert router.snapshot()["tools"]["weather"]["reward_observations"] == 1

        registry.set_enabled("weather", False)
        service.refresh_registry()
        assert retriever.documents == []
        registry.set_enabled("weather", True)
        service.refresh_registry()
        assert router.snapshot()["tools"]["weather"]["reward_observations"] == 1

        assert registry.delete("weather") is True
        router.remove_tool("weather")
        service.refresh_registry()
        assert registry.count() == 0
        assert retriever.documents == []


def test_empty_registry_search_returns_empty_without_calling_encoder():
    class FailingEncoder:
        def encode_many(self, texts):
            raise AssertionError("empty registry must not call embedding provider")

    retriever = SemanticToolRetriever(FailingEncoder(), [])
    router = BudgetAwareToolRouter([], context_dimension=3, diagonal_covariance=True)
    service = ToolRoutingService(retriever, router)
    result = service.search({"query": "anything", "remaining_budget": 1.0})
    assert "recommended_tool" not in result
    assert result["tools"] == []


def test_zero_cost_and_optional_latency():
    from budget_tool_router.registry import definition_assets

    item = definition()
    item.pop("latency")
    _, spec = definition_assets(item)
    assert spec.initial_cost == 0.0
    assert spec.initial_latency is None
    assert spec.prior_count == 1.0
