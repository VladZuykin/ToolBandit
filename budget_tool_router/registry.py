"""Persistent PostgreSQL registry for runtime-managed tool definitions."""

from __future__ import annotations

import json
import time
from typing import Any, Iterable, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .retriever import ToolDocument
from .router import ToolSpec


def definition_assets(item: dict[str, Any]) -> tuple[ToolDocument, ToolSpec]:
    name = str(item["tool_name"])
    value = item.get("latency")
    spec = ToolSpec(
        name=name, initial_cost=float(item["cost"]),
        initial_latency=None if value is None else float(value),
        fixed_cost=None, cost_range=1.0, latency_range=1.0, prior_count=1.0,
    )
    semantic = {"description": item["description"], "input_schema": item["input_schema"],
                "output_schema": item["output_schema"]}
    metadata = {key: value for key, value in item.items()
                if key not in {"cost", "latency", "created_at", "updated_at"}}
    return ToolDocument(name, json.dumps(semantic, ensure_ascii=False), metadata), spec


class ToolRegistry:
    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        self.database_url = database_url
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS tools (
                    tool_name TEXT PRIMARY KEY,
                    definition_json JSONB NOT NULL,
                    enabled BOOLEAN NOT NULL DEFAULT TRUE,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                )
            """)

    @property
    def location(self) -> str:
        return "PostgreSQL"

    def _connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    @staticmethod
    def _row(row: dict[str, Any]) -> dict[str, Any]:
        value = row["definition_json"]
        definition = json.loads(value) if isinstance(value, str) else dict(value)
        definition.update(enabled=bool(row["enabled"]), version=int(row["version"]),
                          created_at=float(row["created_at"]), updated_at=float(row["updated_at"]))
        return definition

    def count(self, *, enabled_only: bool = False) -> int:
        query = "SELECT COUNT(*) AS count FROM tools"
        if enabled_only:
            query += " WHERE enabled = TRUE"
        with self._connect() as connection:
            row = connection.execute(query).fetchone()
        return int(row["count"])

    def list(self, *, limit: int = 100, offset: int = 0,
             enabled: Optional[bool] = None) -> list[dict[str, Any]]:
        query, values = "SELECT * FROM tools", []
        if enabled is not None:
            query += " WHERE enabled = %s"
            values.append(enabled)
        query += " ORDER BY tool_name LIMIT %s OFFSET %s"
        values.extend([limit, offset])
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._row(row) for row in rows]

    def get(self, name: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tools WHERE tool_name=%s", (name,)).fetchone()
        return None if row is None else self._row(row)

    def put(self, definition: dict[str, Any], *, enabled: bool = True,
            create_only: bool = False) -> dict[str, Any]:
        name, now = str(definition["tool_name"]), time.time()
        body = {key: value for key, value in definition.items()
                if key not in {"enabled", "version", "created_at", "updated_at"}}
        with self._connect() as connection:
            if create_only:
                row = connection.execute("""
                    INSERT INTO tools (tool_name, definition_json, enabled, version, created_at, updated_at)
                    VALUES (%s, %s, %s, 1, %s, %s)
                    ON CONFLICT (tool_name) DO NOTHING RETURNING tool_name
                """, (name, Jsonb(body), enabled, now, now)).fetchone()
                if row is None:
                    raise FileExistsError(name)
            else:
                connection.execute("""
                    INSERT INTO tools (tool_name, definition_json, enabled, version, created_at, updated_at)
                    VALUES (%s, %s, %s, 1, %s, %s)
                    ON CONFLICT (tool_name) DO UPDATE SET
                      definition_json=EXCLUDED.definition_json, enabled=EXCLUDED.enabled,
                      version=tools.version+1, updated_at=EXCLUDED.updated_at
                """, (name, Jsonb(body), enabled, now, now))
        result = self.get(name)
        assert result is not None
        return result

    def set_enabled(self, name: str, enabled: bool) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE tools SET enabled=%s, version=version+1, updated_at=%s WHERE tool_name=%s",
                (enabled, time.time(), name),
            )
            if cursor.rowcount == 0:
                return None
        return self.get(name)

    def delete(self, name: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM tools WHERE tool_name=%s", (name,))
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
