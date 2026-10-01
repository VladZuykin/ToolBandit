"""Compare QoSToolBench agentic and baseline benchmark summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ORDER = {
    "agentic": 0,
    "ucb-top1": 1,
    "semantic": 2,
    "random": 3,
    "cheapest": 4,
    "fastest": 5,
    "highest-pass-rate": 6,
    "oracle-utility": 7,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.summaries:
        row = json.loads(path.read_text(encoding="utf-8"))
        row["summary_file"] = str(path)
        rows.append(row)
    rows.sort(key=lambda row: ORDER.get(str(row.get("selection_policy")), 99))

    lines = [
        "| Strategy | Success | Pass rate | Steps | Mean cost | Mean latency | p95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {} | {}/{} | {:.3f} | {:.2f} | {:.6f} | {:.3f} | {:.3f} |".format(
            row.get("selection_policy", "unknown"), row["successes"], row["queries"],
            row["success_rate"], row["mean_steps"], row["mean_cost"],
            row.get("mean_latency") or 0.0, row.get("p95_latency") or 0.0,
        ))
    markdown = "\n".join(lines) + "\n"
    print(markdown, end="")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
