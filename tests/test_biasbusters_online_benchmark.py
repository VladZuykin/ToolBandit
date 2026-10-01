import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_biasbusters_online_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_biasbusters_online_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def api():
    return {
        "category_name": "Mapping",
        "tool_name": "Example Geocoder",
        "api_name": "Forward geocode",
        "api_description": "Find coordinates for an address.",
        "required_parameters": [{"name": "address", "type": "STRING", "default": "Berlin"}],
        "optional_parameters": [],
        "template_response": {"latitude": "float", "longitude": "float"},
    }


def test_api_id_matches_stabletoolbench_normalization():
    assert MODULE.api_id(api()) == "Mapping/example_geocoder/forward_geocode"


def test_catalog_and_arguments_include_query_and_profile():
    item = api()
    name = MODULE.tool_name(item)
    definitions = MODULE.catalog_definitions(
        {name: item},
        {MODULE.api_id(item): {"cost": 0.004, "latency": 1.5}},
    )
    definition = definitions[0]
    assert definition["cost"] == 0.004
    assert definition["latency"] == 1.5
    assert MODULE.default_tool_arguments("Where is Berlin?", definition["metadata"]) == {
        "query": "Where is Berlin?",
        "address": "Berlin",
    }


def test_json_object_accepts_json_code_fence():
    assert MODULE._json_object('```json\n{"address":"Berlin"}\n```') == {"address": "Berlin"}


def test_select_candidate_supports_ucb_top1_and_resource_baselines():
    candidates = [
        {"tool": "a", "ucb_score": 0.4, "retrieval_rank": 1,
         "metadata": {"category_name": "Mapping", "tool_name": "A", "api_name": "one"}},
        {"tool": "b", "ucb_score": 0.8, "retrieval_rank": 2,
         "metadata": {"category_name": "Mapping", "tool_name": "B", "api_name": "two"}},
    ]
    profiles = {
        "Mapping/a/one": {"cost": 0.01, "latency": 0.1, "success_rate": 0.6},
        "Mapping/b/two": {"cost": 0.02, "latency": 0.2, "success_rate": 0.9},
    }
    import random
    kwargs = dict(profiles=profiles, rng=random.Random(42), cost_weight=0.5,
                  latency_weight=0.5, cost_scale=1.0, latency_scale=1.0)
    assert MODULE.select_candidate(candidates, policy="ucb-top1", **kwargs)["tool"] == "b"
    assert MODULE.select_candidate(candidates, policy="cheapest", **kwargs)["tool"] == "a"
    assert MODULE.select_candidate(candidates, policy="fastest", **kwargs)["tool"] == "a"
    assert MODULE.select_candidate(candidates, policy="highest-pass-rate", **kwargs)["tool"] == "b"
