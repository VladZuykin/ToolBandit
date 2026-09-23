import tempfile
from pathlib import Path

import numpy as np

from budget_tool_router.ada_embeddings import Ada002Encoder


class FakeAdaEncoder(Ada002Encoder):
    def _request(self, texts):
        return [np.ones(self.source_dimension, dtype=np.float32) for _ in texts]


def test_sqlite_cache_connections_are_closed_before_temp_cleanup():
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "cache.sqlite3"
        encoder = FakeAdaEncoder(cache)
        first = encoder.encode_many(["connection test"])[0]
        second = encoder.encode_many(["connection test"])[0]
        assert first.shape == (1536,)
        assert np.allclose(first, second)
    assert not Path(directory).exists()
