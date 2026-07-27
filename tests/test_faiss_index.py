from pathlib import Path

import numpy as np

from app.ingestion.faiss_index import FaissIndexManager


def test_faiss_index_save_and_load(tmp_path: Path) -> None:
    index_path = tmp_path / "vectors.faiss"
    manager = FaissIndexManager(index_path, dimension=3)
    manager.add(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), np.array([101], dtype=np.int64))
    manager.save()

    reloaded = FaissIndexManager(index_path, dimension=3)
    scores, ids = reloaded.search(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), limit=1)

    assert reloaded.size == 1
    assert ids[0, 0] == 101
    assert scores[0, 0] == 1.0
