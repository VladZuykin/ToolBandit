from budget_tool_router.adapters import PublicJsonApiAdapter, public_api_examples


def test_showcase_contains_ten_unique_real_api_adapters():
    examples = public_api_examples(timeout=2.0)
    assert len(examples) == 10
    assert len({item.adapter.tool_name for item in examples}) == 10
    for item in examples:
        estimate = item.adapter.estimate(item.arguments)
        assert estimate.estimated_cost == 0.0
        assert estimate.estimated_latency == 2.0


def test_generic_public_adapter_measures_success_and_failure():
    success = PublicJsonApiAdapter(
        tool_name="x", description="x", input_schema={}, output_schema={},
        url_builder=lambda args: "https://example.test/" + args["id"],
        transport=lambda url, timeout, user_agent: (200, {"url": url}),
    )
    result = success.invoke({"id": "42"})
    assert result.status_code == 200
    assert result.output == {"url": "https://example.test/42"}
    assert result.error_code is None

    def broken(url, timeout, user_agent):
        raise TimeoutError("late")

    failure = PublicJsonApiAdapter(
        tool_name="y", description="y", input_schema={}, output_schema={},
        url_builder=lambda args: "https://example.test", transport=broken,
    ).invoke({})
    assert failure.status_code is None
    assert failure.error_code == "TimeoutError"


def test_pokemon_adapter_returns_compact_judge_ready_output():
    example = next(x for x in public_api_examples() if x.adapter.tool_name == "pokeapi_pokemon")
    example.adapter.transport = lambda url, timeout, user_agent: (200, {
        "id": 25, "name": "pikachu", "height": 4, "weight": 60,
        "types": [{"type": {"name": "electric"}}],
        "stats": [{"base_stat": 90, "stat": {"name": "speed"}}],
        "abilities": [{"ability": {"name": "static"}}],
        "sprites": {"huge": "ignored"}, "moves": ["ignored"],
    })
    output = example.adapter.invoke(example.arguments).output
    assert output["types"] == ["electric"]
    assert output["stats"] == {"speed": 90}
    assert "sprites" not in output
    assert "moves" not in output
