import importlib.util
import http.client
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_qostoolbench_agentic_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_qostoolbench_agentic_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_full_catalog_covers_every_qos_profile():
    catalog = ROOT / "external" / "StableToolBenchQoS" / "data" / "catalog" / "tools.jsonl"
    profiles_path = (
        ROOT / "external" / "StableToolBenchQoS" / "data" / "qos" / "v5"
        / "api_qos_profiles.jsonl"
    )
    profiles = MODULE.BASE.load_profiles(profiles_path, "normal")
    apis = MODULE.load_catalog_apis(catalog)
    selected = {
        name: api for name, api in apis.items()
        if MODULE.BASE.api_id(api) in profiles
    }
    assert len(profiles) == 7546
    assert len(selected) == 7546
    assert len(set(selected)) == 7546
    assert max(map(len, selected)) <= 255
    assert len(MODULE.BASE.catalog_definitions(selected, profiles)) == 7546


def test_agent_prompt_explains_all_selection_signals():
    prompt = MODULE.planner_prompt(
        "Generate a QR code",
        [{
            "tool": "qr", "ucb_score": 0.7, "lqm_score": 0.6,
            "retrieval_score": 0.9, "expected_cost": 0.001,
            "expected_latency": 0.2, "input_schema": {}, "metadata": {},
        }],
        [],
    )
    for term in (
        "top-20", "retrieval_score", "ucb_score", "lqm_score",
        "expected_cost", "expected_latency", "do not repeat a failed tool",
    ):
        assert term in prompt


def test_llm_json_retries_incomplete_http_response(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": '{"action":"finish"}'}}]
            }).encode("utf-8")

    calls = 0

    def urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise http.client.IncompleteRead(b"")
        return Response()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setattr(MODULE, "urlopen", urlopen)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)

    assert MODULE.llm_json("test", model="test", timeout=1, retries=1) == {
        "action": "finish"
    }
    assert calls == 2


def test_llm_json_retries_malformed_model_json(monkeypatch):
    class Response:
        def __init__(self, content): self.content = content
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": self.content}}]
            }).encode("utf-8")

    responses = iter([Response('{"action":"call"'), Response('{"action":"finish"}')])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setattr(MODULE, "urlopen", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)

    assert MODULE.llm_json("test", model="test", timeout=1, retries=1) == {
        "action": "finish"
    }
