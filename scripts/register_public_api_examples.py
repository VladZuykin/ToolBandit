"""Idempotently register the ten live showcase tools through ToolBandit's API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from budget_tool_router.adapters import public_api_examples


def request_json(url: str, *, method: str = "GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=180) as response:
            body = response.read()
            return None if not body else json.loads(body.decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP {}: {}".format(error.code, detail)) from error
    except URLError as error:
        raise RuntimeError("ToolBandit is unavailable: {}".format(error.reason)) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    health = request_json(base + "/health")
    print("SERVICE: status={}, active_tools={}".format(health["status"], health["active_tools"]))
    tools = []
    for example in public_api_examples():
        item = example.adapter.catalog_entry()
        item.update({
            "metadata": {
                "provider": "public_api_showcase",
                "example_query": example.query,
                "example_arguments": example.arguments,
                "expected_contract": example.expected_contract,
            },
            "enabled": True,
        })
        tools.append(item)

    print("IMPORT: sending {} tools through POST /v1/tools/import ...".format(len(tools)), flush=True)
    imported = request_json(base + "/v1/tools/import", method="POST", payload={
        "tools": tools, "replace_existing": True,
    })
    print("IMPORT RESULT: " + json.dumps(imported, ensure_ascii=False))

    listing = request_json(base + "/v1/tools?limit=1000&enabled=true")
    registered = {item["tool_name"] for item in listing["items"]}
    expected = {item["tool_name"] for item in tools}
    missing = sorted(expected - registered)
    print("VERIFY: {}/{} expected tools are present".format(len(expected) - len(missing), len(expected)))
    for name in sorted(expected):
        print("  {} {}".format("OK" if name in registered else "MISSING", name))
    if missing:
        raise SystemExit("Registration incomplete: " + ", ".join(missing))


if __name__ == "__main__":
    main()
