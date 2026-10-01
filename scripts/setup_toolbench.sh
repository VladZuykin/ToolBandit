#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_COMMIT="34c5cf0b4a20a7f42e7fd55ef3feeace77d5ff5c"
SUBMODULE="$ROOT/external/StableToolBenchQoS"
VENV="$ROOT/.venv-toolbench-server"

git -C "$ROOT" submodule update --init --recursive external/StableToolBenchQoS
ACTUAL_COMMIT="$(git -C "$SUBMODULE" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "StableToolBench revision mismatch: expected $EXPECTED_COMMIT, got $ACTUAL_COMMIT" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
"$PYTHON_BIN" -m venv "$VENV"
"$VENV/Scripts/python.exe" -m pip install --upgrade pip
"$VENV/Scripts/python.exe" -m pip install -r "$ROOT/requirements-toolbench-server.lock.txt"
"$VENV/Scripts/python.exe" "$ROOT/scripts/check_toolbench_setup.py"

echo "StableToolBench server environment is ready: $VENV"
