"""Compare BiasBusters online benchmark summary JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ORDER = {
    "ucb-top1": 0,
    "ucb": 0,
    "lqm-context-route": 1,
    "semantic": 2,
    "random": 3,
    "cheapest": 4,
    "fastest": 5,
    "highest-pass-rate": 6,
    "oracle-utility": 7,
}


def label(row: dict[str, Any]) -> str:
    policy = row.get("selection_policy", "unknown")
    parameters = row.get("router_parameters") or {}
    if policy in {"ucb", "ucb-top1"}:
        return "ucb-top1(c={},l={},alpha={})".format(
            parameters.get("cost_weight"), parameters.get("latency_weight"),
            parameters.get("alpha"),
        )
    return str(policy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.summaries:
        row = json.loads(path.read_text(encoding="utf-8"))
        row["summary_file"] = str(path)
        rows.append(row)
    rows.sort(key=lambda row: (ORDER.get(row.get("selection_policy"), 99), label(row)))
    lines = [
        "| Strategy | Success | Pass rate | Attempts | Mean cost | Mean latency | p95 latency | Judge failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {} | {}/{} | {:.3f} | {:.2f} | {:.6f} | {:.3f} | {:.3f} | {} |".format(
                label(row), row["successes"], row["queries"], row["success_rate"],
                row["mean_attempts"], row["mean_cost"], row["mean_latency"],
                row["p95_latency"], row["judge_failures"],
            )
        )
    rendered = "\n".join(lines) + "\n"
    print(rendered, end="")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
