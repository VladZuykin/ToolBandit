"""Offline-first preflight checks for starting ToolBandit."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from budget_tool_router.registry import ToolRegistry


def result(ok: bool, label: str, detail: str) -> int:
    print("{} {:<18} {}".format("OK  " if ok else "FAIL", label, detail))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ToolBandit before startup")
    parser.add_argument("--registry", default=os.getenv(
        "TOOL_REGISTRY_PATH", str(PROJECT_ROOT / "data" / "tool_registry.sqlite3")
    ))
    args = parser.parse_args()
    failures = 0

    version_ok = sys.version_info >= (3, 11)
    failures += result(version_ok, "Python", sys.version.split()[0] + " (need >=3.11)")
    for package in ("numpy", "fastapi", "uvicorn", "pydantic"):
        try:
            module = importlib.import_module(package)
            failures += result(True, package, getattr(module, "__version__", "installed"))
        except Exception as error:
            failures += result(False, package, str(error))

    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        present = bool(os.getenv(name))
        note = "set" if present else "missing"
        failures += result(present, name, note)

    path = Path(args.registry).expanduser().resolve()
    try:
        registry = ToolRegistry(path)
        failures += result(True, "Tool registry", "{} tools in {}".format(registry.count(), path))
        failures += result(True, "Enabled tools", str(registry.count(enabled_only=True)))
    except Exception as error:
        failures += result(False, "Tool registry", str(error))

    print("\n{}".format("READY" if not failures else "NOT READY: {} required check(s) failed".format(failures)))
    print("For paid-provider network calls run: python check_api_connections.py")
    print("For ten public API calls run: python run_real_api_showcase.py --skip-judge")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
