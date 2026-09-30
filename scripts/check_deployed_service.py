"""Smoke-test an already deployed ToolBandit API and its external integrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            return {} if not body else json.loads(body.decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP {} {}: {}".format(error.code, url, detail)) from error
    except URLError as error:
        raise RuntimeError("cannot reach {}: {}".format(url, error.reason)) from error


def call_open_meteo(*, timeout: float = 15.0) -> tuple[dict[str, Any], float]:
    query = urlencode({
        "latitude": 55.7558,
        "longitude": 37.6173,
        "current": "temperature_2m,weather_code",
    })
    started = time.perf_counter()
    try:
        with urlopen(
            Request(
                "https://api.open-meteo.com/v1/forecast?" + query,
                headers={"Accept": "application/json", "User-Agent": "ToolBandit-smoke-test/1.0"},
            ),
            timeout=timeout,
        ) as response:
            output = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError("Open-Meteo call failed: {}".format(error)) from error
    return output, time.perf_counter() - started


def run_smoke(
    base_url: str,
    *,
    judge_timeout: float = 120.0,
    keep_tool: bool = False,
    api_request: Callable[..., dict[str, Any]] = request_json,
    tool_call: Callable[..., tuple[dict[str, Any], float]] = call_open_meteo,
) -> dict[str, Any]:
    base = base_url.rstrip("/")
    suffix = uuid.uuid4().hex[:12]
    tool_name = "toolbandit_smoke_weather_{}".format(suffix)
    created = False
    summary: dict[str, Any] = {"tool": tool_name}

    print("[1/7] HEALTH: checking deployed FastAPI ...", flush=True)
    health = api_request(base + "/health")
    if health.get("status") != "ok":
        raise RuntimeError("service is not healthy: {}".format(health))
    if not health.get("registry"):
        raise RuntimeError("deployed service has no tool registry")
    if not health.get("judge"):
        raise RuntimeError("deployed service has no Judge")
    print("      OK: active_tools={}, covariance={}".format(
        health.get("active_tools"), health.get("covariance")
    ))

    definition = {
        "tool_name": tool_name,
        "description": (
            "ToolBandit deployment smoke test: get current weather by latitude "
            "and longitude from Open-Meteo."
        ),
        "input_schema": {
            "type": "object",
            "required": ["latitude", "longitude"],
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
            },
        },
        "output_schema": {
            "type": "object",
            "required": ["latitude", "longitude", "current"],
        },
        "cost": 0.0,
        "metadata": {"provider": "open-meteo", "purpose": "deployment-smoke-test"},
        "enabled": True,
    }

    try:
        print("[2/7] REGISTRY: registering a temporary tool ...", flush=True)
        imported = api_request(
            base + "/v1/tools/import",
            method="POST",
            payload={"tools": [definition], "replace_existing": False},
        )
        if int(imported.get("imported", 0)) != 1:
            raise RuntimeError("temporary tool was not imported: {}".format(imported))
        created = True
        stored = api_request(base + "/v1/tools/" + tool_name)
        if stored.get("tool_name") != tool_name:
            raise RuntimeError("registered tool cannot be read back")
        print("      OK: {}".format(tool_name))

        print("[3/7] SEARCH: semantic retrieval + budget/SLA filtering + UCB ...", flush=True)
        search_payload = {
            "query": definition["description"],
            "remaining_budget": 1.0,
            "latency_sla": 15.0,
            "retrieval_limit": 1000,
            "result_limit": 1000,
            "retrieval_threshold": -1.0,
        }
        search = api_request(base + "/v1/tools/search", method="POST", payload=search_payload)
        selected = next((row for row in search.get("tools", []) if row.get("tool") == tool_name), None)
        if selected is None:
            raise RuntimeError("temporary tool was not returned by search")
        initial_score = float(selected["ucb_score"])
        print("      OK: retrieval_rank={}, ucb_score={:.6f}".format(
            selected.get("retrieval_rank"), initial_score
        ))

        print("[4/7] TOOL CALL: invoking real Open-Meteo API ...", flush=True)
        output, observed_latency = tool_call(timeout=15.0)
        current = output.get("current")
        if not isinstance(current, dict) or not isinstance(current.get("temperature_2m"), (int, float)):
            raise RuntimeError("Open-Meteo returned an unexpected response")
        print("      OK: temperature={}, latency={:.3f}s".format(
            current["temperature_2m"], observed_latency
        ))

        print("[5/7] FEEDBACK: submitting the real result to asynchronous Judge ...", flush=True)
        interaction_id = "smoke-{}".format(suffix)
        job = api_request(
            base + "/v1/evaluations",
            method="POST",
            payload={
                "interaction_id": interaction_id,
                "request_id": search["request_id"],
                "tool_name": tool_name,
                "tool_intent": "Get the current weather for Moscow coordinates.",
                "expected_contract": (
                    "Return latitude, longitude, and current with numeric temperature_2m."
                ),
                "tool_output": output,
                "observed_cost": 0.0,
                "observed_latency": observed_latency,
            },
        )
        evaluation_id = str(job["evaluation_id"])
        print("      ACCEPTED: evaluation_id={}".format(evaluation_id))

        print("[6/7] JUDGE: waiting for delayed reward ...", flush=True)
        deadline = time.monotonic() + judge_timeout
        while True:
            job = api_request(base + "/v1/evaluations/" + evaluation_id)
            if job.get("status") != "pending":
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Judge did not finish within {:.0f}s".format(judge_timeout))
            time.sleep(0.25)
        if job.get("status") != "completed":
            raise RuntimeError("Judge failed: {}".format(job.get("error", job)))
        judgment = job.get("judgment", {})
        if not job.get("learned"):
            raise RuntimeError("Judge returned an uncertain verdict; LinUCB was not updated")
        print("      OK: verdict={}, confidence={}, learned={}".format(
            judgment.get("verdict"), judgment.get("confidence"), job.get("learned")
        ))

        print("[7/7] LEARNING: searching again after the LinUCB update ...", flush=True)
        after = api_request(base + "/v1/tools/search", method="POST", payload=search_payload)
        updated = next((row for row in after.get("tools", []) if row.get("tool") == tool_name), None)
        if updated is None:
            raise RuntimeError("temporary tool disappeared after evaluation")
        updated_score = float(updated["ucb_score"])
        if updated_score == initial_score:
            raise RuntimeError("ucb_score did not change after learned feedback")
        print("      OK: ucb_score {:.6f} -> {:.6f}".format(initial_score, updated_score))

        summary.update({
            "status": "ok",
            "judge_verdict": judgment.get("verdict"),
            "initial_ucb_score": initial_score,
            "updated_ucb_score": updated_score,
            "observed_latency": observed_latency,
        })
        return summary
    finally:
        if created and not keep_tool:
            try:
                api_request(base + "/v1/tools/" + tool_name, method="DELETE")
                print("CLEANUP: temporary tool deleted")
            except Exception as error:
                print("CLEANUP WARNING: {}".format(error), file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test a deployed ToolBandit API, PostgreSQL registry, real tool and Judge."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--judge-timeout", type=float, default=120.0)
    parser.add_argument("--keep-tool", action="store_true")
    args = parser.parse_args()
    try:
        result = run_smoke(
            args.base_url,
            judge_timeout=args.judge_timeout,
            keep_tool=args.keep_tool,
        )
        print("\nRESULT\n" + json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print("\nFAILED: {}: {}".format(type(error).__name__, error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
