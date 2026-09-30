import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_deployed_service.py"
SPEC = importlib.util.spec_from_file_location("check_deployed_service", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_deployed_smoke_covers_registry_search_tool_judge_and_cleanup(monkeypatch):
    state = {"tool": None, "score": 0.35, "deleted": False}

    def api_request(url, *, method="GET", payload=None, timeout=180.0):
        del timeout
        if url.endswith("/health"):
            return {"status": "ok", "registry": True, "judge": True,
                    "active_tools": 0, "covariance": "diagonal"}
        if url.endswith("/v1/tools/import"):
            state["tool"] = payload["tools"][0]["tool_name"]
            return {"imported": 1, "active_tools": 1, "total": 1}
        if url.endswith("/v1/tools/search"):
            return {"request_id": "request-1", "tools": [{
                "tool": state["tool"], "ucb_score": state["score"],
                "retrieval_rank": 1,
            }]}
        if url.endswith("/v1/evaluations") and method == "POST":
            return {"evaluation_id": "evaluation-1", "status": "pending"}
        if url.endswith("/v1/evaluations/evaluation-1"):
            state["score"] = 0.7
            return {"evaluation_id": "evaluation-1", "status": "completed",
                    "learned": True, "judgment": {
                        "verdict": "pass", "confidence": 1.0,
                    }}
        if state["tool"] and url.endswith("/v1/tools/" + state["tool"]):
            if method == "DELETE":
                state["deleted"] = True
                return {}
            return {"tool_name": state["tool"]}
        raise AssertionError("unexpected request: {} {}".format(method, url))

    def tool_call(*, timeout):
        assert timeout == 15.0
        return {
            "latitude": 55.7558,
            "longitude": 37.6173,
            "current": {"temperature_2m": 12.0},
        }, 0.25

    result = MODULE.run_smoke(
        "http://service:8080", api_request=api_request, tool_call=tool_call
    )

    assert result["status"] == "ok"
    assert result["initial_ucb_score"] == 0.35
    assert result["updated_ucb_score"] == 0.7
    assert state["deleted"] is True
