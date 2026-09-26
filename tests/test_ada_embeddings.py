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
