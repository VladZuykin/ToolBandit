"""Cached OpenAI text-embedding-ada-002 embeddings."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
from contextlib import closing
from typing import Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np


class Ada002Encoder:
    model = "text-embedding-ada-002"
    source_dimension = 1536

    def __init__(
        self,
        cache_path: str | Path,
        *,
        output_dimension: int = 1536,
        projection_seed: int = 42,
        batch_size: int = 100,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.output_dimension = output_dimension
        self.batch_size = batch_size
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        if output_dimension <= 0 or output_dimension > self.source_dimension:
            raise ValueError("output_dimension must be in [1, 1536]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.projection = None
        if output_dimension != self.source_dimension:
            rng = np.random.default_rng(projection_seed)
            self.projection = rng.normal(
                0.0,
                1.0 / np.sqrt(output_dimension),
                size=(self.source_dimension, output_dimension),
            ).astype(np.float32)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_cache()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.cache_path)

    def _initialize_cache(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings "
                "(cache_key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )
            connection.commit()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _read_cached(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        with closing(self._connect()) as connection:
            for text in texts:
                row = connection.execute(
                    "SELECT vector FROM embeddings WHERE cache_key = ?",
                    (self._key(text),),
                ).fetchone()
                if row is not None:
                    result[text] = np.frombuffer(row[0], dtype=np.float32).copy()
        return result

    def _request(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set and the embedding cache is missing "
                f"{len(texts)} requested text(s)"
            )
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        request = Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI embeddings request failed ({error.code}): {detail}") from error
        ordered = sorted(body["data"], key=lambda item: item["index"])
        vectors = [np.asarray(item["embedding"], dtype=np.float32) for item in ordered]
        if any(vector.shape != (self.source_dimension,) for vector in vectors):
            raise RuntimeError("unexpected ada-002 embedding dimension")
        return vectors

    def _project(self, vectors: Sequence[np.ndarray]) -> list[np.ndarray]:
        matrix = np.vstack(vectors)
        projected = matrix if self.projection is None else matrix @ self.projection
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        projected = projected / np.maximum(norms, 1e-12)
        return [row.astype(np.float64) for row in projected]

    def encode_many(self, texts: Sequence[str]) -> list[np.ndarray]:
        unique = list(dict.fromkeys(texts))
        cached = self._read_cached(unique)
        missing = [text for text in unique if text not in cached]

        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            source_vectors = self._request(batch)
            projected = self._project(source_vectors)
            with closing(self._connect()) as connection:
                for text, vector in zip(batch, projected):
                    connection.execute(
                        "INSERT OR REPLACE INTO embeddings(cache_key, vector) VALUES (?, ?)",
                        (self._key(text), vector.astype(np.float32).tobytes()),
                    )
                    cached[text] = vector
                connection.commit()
            print(f"Embedded {min(start + len(batch), len(missing))}/{len(missing)} uncached texts")

        return [cached[text] for text in texts]
