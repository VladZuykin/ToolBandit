"""Offline-first preflight checks for starting ToolBandit."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def result(ok: bool, label: str, detail: str) -> int:
    print("{} {:<18} {}".format("OK  " if ok else "FAIL", label, detail))
    return 0 if ok else 1


def main() -> int:
    failures = 0
    dependencies_ok = True

    version_ok = sys.version_info >= (3, 11)
    failures += result(version_ok, "Python", sys.version.split()[0] + " (need >=3.11)")
    for package in ("numpy", "fastapi", "uvicorn", "pydantic", "psycopg"):
        try:
            module = importlib.import_module(package)
            failures += result(True, package, getattr(module, "__version__", "installed"))
        except Exception as error:
            dependencies_ok = False
            failures += result(False, package, str(error))

    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        present = bool(os.getenv(name))
        note = "set" if present else "missing"
        failures += result(present, name, note)

    database_url = os.getenv("DATABASE_URL")
    if dependencies_ok:
        try:
            # Import only after dependency checks so preflight can report a
            # missing package instead of crashing during module initialization.
            from budget_tool_router.registry import ToolRegistry

            registry = ToolRegistry(database_url or "")
            failures += result(True, "Tool registry", "{} tools in PostgreSQL".format(registry.count()))
            failures += result(True, "Enabled tools", str(registry.count(enabled_only=True)))
        except Exception as error:
            failures += result(False, "Tool registry", str(error))
    else:
        failures += result(False, "Tool registry", "skipped: install dependencies first")

    print("\n{}".format("READY" if not failures else "NOT READY: {} required check(s) failed".format(failures)))
    print("For paid-provider network calls run: python scripts/check_api_connections.py")
    print("For ten public API calls run: python scripts/run_real_api_showcase.py --skip-judge")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
