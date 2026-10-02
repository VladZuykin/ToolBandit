import numpy as np

from budget_tool_router import ada_embeddings
from budget_tool_router.ada_embeddings import (
    Ada002Encoder, MemoryEmbeddingCache, PostgresEmbeddingCache,
)


class FakeAdaEncoder(Ada002Encoder):
    def _request(self, texts):
        return [np.ones(self.source_dimension, dtype=np.float32) for _ in texts]


def test_embedding_cache_avoids_duplicate_provider_calls():
    encoder = FakeAdaEncoder(cache=MemoryEmbeddingCache())
    first = encoder.encode_many(["connection test"])[0]
    second = encoder.encode_many(["connection test"])[0]
    assert first.shape == (1536,)
    assert np.allclose(first, second)


def test_embedding_batch_is_split_when_provider_context_limit_is_exceeded():
    class ContextLimitedEncoder(FakeAdaEncoder):
        def __init__(self):
            super().__init__(cache=MemoryEmbeddingCache(), batch_size=4)
            self.request_sizes = []

        def _request(self, texts):
            self.request_sizes.append(len(texts))
            if len(texts) > 2:
                raise RuntimeError(
                    "OpenAI embeddings request failed (400): maximum context length"
                )
            return super()._request(texts)

    encoder = ContextLimitedEncoder()
    vectors = encoder.encode_many(["one", "two", "three", "four"])

    assert len(vectors) == 4
    assert encoder.request_sizes == [4, 2, 2]


def test_single_oversized_embedding_text_is_adaptively_truncated():
    class LengthLimitedEncoder(FakeAdaEncoder):
        def __init__(self):
            super().__init__(cache=MemoryEmbeddingCache())
            self.request_lengths = []

        def _request(self, texts):
            self.request_lengths.append(len(texts[0]))
            if len(texts[0]) > 100:
                raise RuntimeError(
                    "OpenAI embeddings request failed (400): maximum context length"
                )
            return super()._request(texts)

    encoder = LengthLimitedEncoder()
    vector = encoder.encode_many(["x" * 200])[0]

    assert vector.shape == (1536,)
    assert encoder.request_lengths[0] == 200
    assert encoder.request_lengths[-1] <= 100
    assert len(encoder.request_lengths) > 1


def test_postgres_cache_writes_through_cursor(monkeypatch):
    class Cursor:
        def __init__(self):
            self.rows = None
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def executemany(self, _query, rows): self.rows = rows

    class Connection:
        def __init__(self): self.db_cursor = Cursor()
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def execute(self, _query): return None
        def cursor(self): return self.db_cursor

    connections = []
    def connect(_url):
        connection = Connection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(ada_embeddings.psycopg, "connect", connect)
    cache = PostgresEmbeddingCache("postgresql://test")
    cache.write([("hello", np.ones(1536, dtype=np.float32))])
    assert len(connections) == 2
    assert len(connections[1].db_cursor.rows) == 1
