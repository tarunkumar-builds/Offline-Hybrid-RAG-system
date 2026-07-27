from pathlib import Path

from app.utils.hashing import sha256_file


def test_sha256_file_is_content_based(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"identical PDF bytes")
    second.write_bytes(b"identical PDF bytes")

    assert sha256_file(first) == sha256_file(second)
