import numpy as np

from budget_tool_router import BudgetAwareToolRouter, SemanticToolRetriever, ToolRoutingService


class FakeEncoder:
    def encode_many(self, texts):
        return [np.array([1.0, 0.0, 0.0]) for _ in texts]


def definition(name="weather", description="Current weather"):
    return {
        "tool_name": name, "description": description,
        "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
        "cost": 0.0, "latency": 0.5, "metadata": {},
    }


class MemoryRegistry:
    def __init__(self):
        self.items = {}
    def put(self, item, **_): self.items[item["tool_name"]] = dict(item)
    def list(self, **_): return [item for item in self.items.values() if item.get("enabled", True)]
    def set_enabled(self, name, enabled): self.items[name]["enabled"] = enabled
    def delete(self, name): return self.items.pop(name, None) is not None
    def count(self): return len(self.items)


def test_registry_refresh_preserves_learning_on_update():
    registry = MemoryRegistry()
    registry.put(definition())
    retriever = SemanticToolRetriever(FakeEncoder(), [])
    router = BudgetAwareToolRouter([], context_dimension=3, diagonal_covariance=True)
    service = ToolRoutingService(retriever, router, registry=registry)
    assert service.refresh_registry() == 1
    router.observe("weather", np.array([1.0, 0.0, 0.0]), passed=True,
                   observed_cost=0.0, observed_latency=0.4)
    registry.put(definition(description="Updated weather description"))
    service.refresh_registry()
    assert router.snapshot()["tools"]["weather"]["reward_observations"] == 1
    registry.set_enabled("weather", False)
    service.refresh_registry()
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
