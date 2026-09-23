"""Semantic retrieval over tool descriptions and schemas."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np


class Encoder(Protocol):
    def encode_many(self, texts: Sequence[str]) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class ToolDocument:
    tool_name: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievalHit:
    tool_name: str
    retrieval_score: float
    retrieval_rank: int
    metadata: dict[str, Any]


def toolbench_documents(rows: Iterable[dict]) -> list[ToolDocument]:
    documents: dict[str, ToolDocument] = {}
    for row in rows:
        for api in row["api_list"]:
            name = f"{api['tool_name']}::{api['api_name']}"
            if name in documents:
                continue
            required = api.get("required_parameters", [])
            optional = api.get("optional_parameters", [])
            response = api.get("template_response")
            text = "\n".join(
                [
                    f"Category: {api.get('category_name', '')}",
                    f"Tool: {api.get('tool_name', '')}",
                    f"Operation: {api.get('api_name', '')}",
                    f"Description: {api.get('api_description', '')}",
                    f"Method: {api.get('method', '')}",
                    "Required parameters: " + json.dumps(required, ensure_ascii=False),
                    "Optional parameters: " + json.dumps(optional, ensure_ascii=False),
                    "Output schema: " + json.dumps(response, ensure_ascii=False),
                ]
            )
            documents[name] = ToolDocument(name, text, dict(api))
    return list(documents.values())


class SemanticToolRetriever:
    def __init__(self, encoder: Encoder, documents: Sequence[ToolDocument]) -> None:
        self.encoder = encoder
        self.documents: list[ToolDocument] = []
        self.matrix = np.empty((0, 0), dtype=np.float64)
        self.replace_documents(documents)

    def replace_documents(self, documents: Sequence[ToolDocument]) -> None:
        """Atomically rebuild the searchable catalog (encoder cache is reused)."""
        new_documents = list(documents)
        if not new_documents:
            self.documents = []
            self.matrix = np.empty((0, 0), dtype=np.float64)
            return
        vectors = np.vstack(self.encoder.encode_many([item.text for item in new_documents])).astype(np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("tool encoder returned a zero vector")
        new_matrix = vectors / norms
        self.documents = new_documents
        self.matrix = new_matrix

    def encode_query(self, query: str) -> np.ndarray:
        vector = np.asarray(self.encoder.encode_many([query])[0], dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("query encoder returned a zero vector")
        return vector / norm

    def search_vector(
        self, vector: Sequence[float] | np.ndarray, *, limit: int = 20,
        threshold: float = 0.0,
    ) -> list[RetrievalHit]:
        if limit <= 0:
            return []
        if not self.documents:
            return []
        query = np.asarray(vector, dtype=np.float64).reshape(-1)
        if query.shape[0] != self.matrix.shape[1]:
            raise ValueError("query and tool embedding dimensions differ")
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores = self.matrix @ query
        order = np.argsort(-scores, kind="stable")
        hits = []
        for index in order:
            score = float(scores[index])
            if score < threshold:
                continue
            doc = self.documents[int(index)]
            hits.append(RetrievalHit(doc.tool_name, score, len(hits) + 1, doc.metadata))
            if len(hits) >= limit:
                break
        return hits

    def search(self, query: str, *, limit: int = 20, threshold: float = 0.0):
        vector = self.encode_query(query)
        return vector, self.search_vector(vector, limit=limit, threshold=threshold)
