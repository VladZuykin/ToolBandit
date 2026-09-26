"""Validate the pinned StableToolBench checkout and a BiasBusters dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "toolbench.lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dataset(path: Path) -> dict[str, int | str]:
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset must be a non-empty JSON array")
    unique_tools: set[tuple[str, str]] = set()
    unique_queries: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} is not an object")
        for field in ("query", "api_list", "relevant APIs", "query_id"):
            if field not in row:
                raise ValueError(f"row {index} has no {field!r}")
        if len(row["api_list"]) != 5:
            raise ValueError(f"row {index} has {len(row['api_list'])} APIs instead of 5")
        unique_queries.add(str(row["query"]))
        for api in row["api_list"]:
            unique_tools.add((str(api["tool_name"]), str(api["api_name"])))
    return {
        "records": len(rows),
        "unique_queries": len(unique_queries),
        "unique_tools": len(unique_tools),
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--allow-different-dataset", action="store_true")
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    checkout = ROOT / lock["path"]
    if not checkout.is_dir():
        raise SystemExit("StableToolBench submodule is missing; run git submodule update --init --recursive")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != lock["commit"]:
        raise SystemExit(f"StableToolBench revision mismatch: {actual_commit}")
    print(f"StableToolBench: OK ({actual_commit})")
    print(f"Python: {sys.version.split()[0]}")

    if args.dataset is None:
        return
    dataset = args.dataset.expanduser().resolve()
    stats = inspect_dataset(dataset)
    expected = lock["biasbusters_dataset"]
    if not args.allow_different_dataset:
        if stats["sha256"] != expected["sha256"]:
            raise SystemExit("BiasBusters dataset checksum differs from toolbench.lock.json")
        if stats["records"] != expected["records"]:
            raise SystemExit("BiasBusters dataset record count differs from toolbench.lock.json")
    print("BiasBusters dataset: OK")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
