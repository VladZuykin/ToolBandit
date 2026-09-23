"""Persistent SQLite registry for runtime-managed tool definitions."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable, Optional

from .retriever import ToolDocument
from .router import ToolSpec


def definition_assets(item: dict[str, Any]) -> tuple[ToolDocument, ToolSpec]:
    """Convert one persisted API definition into retrieval and bandit objects."""
    name = str(item["tool_name"])
    initial_cost = float(item["cost"])
    value = item.get("latency")
    initial_latency = None if value is None else float(value)
    spec = ToolSpec(
        name=name, initial_cost=initial_cost, initial_latency=initial_latency,
        fixed_cost=None, cost_range=1.0, latency_range=1.0, prior_count=1.0,
    )
    semantic = {
        "description": item["description"],
        "input_schema": item["input_schema"], "output_schema": item["output_schema"],
    }
    metadata = {key: value for key, value in item.items()
                if key not in {"cost", "latency", "created_at", "updated_at"}}
    return ToolDocument(name, json.dumps(semantic, ensure_ascii=False), metadata), spec


class ToolRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS tools (
                    tool_name TEXT PRIMARY KEY,
                    definition_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            connection.commit()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _row(row) -> dict[str, Any]:
        definition = json.loads(row["definition_json"])
        # Read legacy registry rows through the current compact API schema.
        if "cost" not in definition and "pricing" in definition:
            pricing = definition.pop("pricing")
            definition["cost"] = float(
                pricing.get("cost", pricing.get("historical_mean", 0.0))
            )
        latency = definition.get("latency")
        if isinstance(latency, dict):
            value = latency.get("historical_mean")
            if value is None:
                definition.pop("latency", None)
            else:
                definition["latency"] = float(value)
        definition.pop("historical_observations", None)
        definition.pop("tags", None)
        definition.update(enabled=bool(row["enabled"]), version=int(row["version"]),
                          created_at=float(row["created_at"]), updated_at=float(row["updated_at"]))
        return definition

    def count(self, *, enabled_only: bool = False) -> int:
        where = " WHERE enabled = 1" if enabled_only else ""
        with closing(self._connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM tools" + where).fetchone()[0])

    def list(self, *, limit: int = 100, offset: int = 0,
             enabled: Optional[bool] = None) -> list[dict[str, Any]]:
        where, values = "", []
        if enabled is not None:
            where, values = " WHERE enabled = ?", [int(enabled)]
        values.extend([limit, offset])
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM tools" + where + " ORDER BY tool_name LIMIT ? OFFSET ?", values
            ).fetchall()
        return [self._row(row) for row in rows]

    def get(self, name: str) -> Optional[dict[str, Any]]:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM tools WHERE tool_name = ?", (name,)).fetchone()
        return None if row is None else self._row(row)

    def put(self, definition: dict[str, Any], *, enabled: bool = True,
            create_only: bool = False) -> dict[str, Any]:
        name = str(definition["tool_name"])
        now = time.time()
        body = {key: value for key, value in definition.items()
                if key not in {"enabled", "version", "created_at", "updated_at"}}
        serialized = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        with closing(self._connect()) as connection:
            existing = connection.execute(
                "SELECT version, created_at FROM tools WHERE tool_name = ?", (name,)
            ).fetchone()
            if existing is not None and create_only:
                raise FileExistsError(name)
            if existing is None:
                connection.execute(
                    "INSERT INTO tools VALUES (?, ?, ?, 1, ?, ?)",
                    (name, serialized, int(enabled), now, now),
                )
            else:
                connection.execute(
                    "UPDATE tools SET definition_json=?, enabled=?, version=?, updated_at=? WHERE tool_name=?",
                    (serialized, int(enabled), int(existing["version"]) + 1, now, name),
                )
            connection.commit()
        return self.get(name)

    def set_enabled(self, name: str, enabled: bool) -> Optional[dict[str, Any]]:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE tools SET enabled=?, version=version+1, updated_at=? WHERE tool_name=?",
                (int(enabled), time.time(), name),
            )
            connection.commit()
            if cursor.rowcount == 0:
                return None
        return self.get(name)

    def delete(self, name: str) -> bool:
        with closing(self._connect()) as connection:
            cursor = connection.execute("DELETE FROM tools WHERE tool_name = ?", (name,))
            connection.commit()
            return cursor.rowcount > 0

    def import_many(self, definitions: Iterable[dict[str, Any]], *, replace: bool = False) -> int:
        count = 0
        for definition in definitions:
            try:
                self.put(definition, enabled=bool(definition.get("enabled", True)), create_only=not replace)
                count += 1
            except FileExistsError:
                continue
        return count
