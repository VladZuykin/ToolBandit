"""OpenAI text-embedding-ada-002 embeddings cached in PostgreSQL."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Protocol, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import psycopg
from psycopg import Binary


class EmbeddingCache(Protocol):
    def read(self, texts: Sequence[str]) -> dict[str, np.ndarray]: ...
    def write(self, values: Sequence[tuple[str, np.ndarray]]) -> None: ...


class PostgresEmbeddingCache:
    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        self.database_url = database_url
        with psycopg.connect(database_url) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    vector BYTEA NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def read(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        by_key = {self._key(text): text for text in texts}
        if not by_key:
            return {}
        with psycopg.connect(self.database_url) as connection:
            rows = connection.execute(
                "SELECT cache_key, vector FROM embeddings WHERE cache_key = ANY(%s)",
                (list(by_key),),
            ).fetchall()
        return {by_key[key]: np.frombuffer(bytes(vector), dtype=np.float32).copy()
                for key, vector in rows}

    def write(self, values: Sequence[tuple[str, np.ndarray]]) -> None:
        rows = [(self._key(text), Binary(vector.astype(np.float32).tobytes()))
                for text, vector in values]
        if not rows:
            return
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.executemany("""
                    INSERT INTO embeddings (cache_key, vector) VALUES (%s, %s)
                    ON CONFLICT (cache_key) DO UPDATE SET vector=EXCLUDED.vector
                """, rows)


class MemoryEmbeddingCache:
    """Non-persistent cache intended for tests."""
    def __init__(self) -> None:
        self.values: dict[str, np.ndarray] = {}

    def read(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        return {text: self.values[text].copy() for text in texts if text in self.values}

    def write(self, values: Sequence[tuple[str, np.ndarray]]) -> None:
        for text, vector in values:
            self.values[text] = vector.copy()


class Ada002Encoder:
    model = "text-embedding-ada-002"
    source_dimension = 1536

    def __init__(self, database_url: str | None = None, *, cache: EmbeddingCache | None = None,
                 output_dimension: int = 1536, projection_seed: int = 42,
                 batch_size: int = 100) -> None:
        self.output_dimension, self.batch_size = output_dimension, batch_size
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        if output_dimension <= 0 or output_dimension > self.source_dimension:
            raise ValueError("output_dimension must be in [1, 1536]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.cache = cache or PostgresEmbeddingCache(database_url or "")
        self.projection = None
        if output_dimension != self.source_dimension:
            rng = np.random.default_rng(projection_seed)
            self.projection = rng.normal(0.0, 1.0 / np.sqrt(output_dimension),
                size=(self.source_dimension, output_dimension)).astype(np.float32)

    def _request(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set and the embedding cache is missing "
                               f"{len(texts)} requested text(s)")
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        request = Request("https://api.openai.com/v1/embeddings", data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST")
        try:
            with urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI embeddings request failed ({error.code}): {detail}") from error
        vectors = [np.asarray(item["embedding"], dtype=np.float32)
                   for item in sorted(body["data"], key=lambda item: item["index"])]
        if any(vector.shape != (self.source_dimension,) for vector in vectors):
            raise RuntimeError("unexpected ada-002 embedding dimension")
        return vectors

    def _project(self, vectors: Sequence[np.ndarray]) -> list[np.ndarray]:
        matrix = np.vstack(vectors)
        projected = matrix if self.projection is None else matrix @ self.projection
        projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
        return [row.astype(np.float64) for row in projected]

    def encode_many(self, texts: Sequence[str]) -> list[np.ndarray]:
        unique = list(dict.fromkeys(texts))
        cached = self.cache.read(unique)
        missing = [text for text in unique if text not in cached]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            projected = self._project(self._request(batch))
            self.cache.write(list(zip(batch, projected)))
            cached.update(zip(batch, projected))
            print(f"Embedded {min(start + len(batch), len(missing))}/{len(missing)} uncached texts")
        return [cached[text] for text in texts]
