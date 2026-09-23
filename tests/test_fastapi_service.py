from types import SimpleNamespace

from budget_tool_router import create_app


def test_fastapi_routes_and_openapi_are_exposed():
    service = SimpleNamespace(
        judge=None, registry=None,
        router=SimpleNamespace(tool_names=[], diagonal_covariance=True),
    )
    app = create_app(service)
    paths = app.openapi()["paths"]
    assert "/health" in paths
    assert "/v1/tools/search" in paths
    assert "/v1/tools" in paths
    assert "/v1/tools/{tool_name}" in paths
    assert "/v1/tools/{tool_name}/enabled" in paths
    assert "/v1/tools/import" in paths
    assert "/v1/registry/stats" in paths
    assert "/v1/evaluations" in paths
    assert "/v1/evaluations/{evaluation_id}" in paths
    assert app.title == "ToolBandit API"
    health_route = next(route for route in app.routes if getattr(route, "path", None) == "/health")
    health = health_route.endpoint()
    assert health["ranking_policy"] == "semantic_order_with_ucb_annotation"
    assert health["covariance"] == "diagonal"
