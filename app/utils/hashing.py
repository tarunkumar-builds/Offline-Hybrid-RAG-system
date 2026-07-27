"""Content hashing helpers."""

from hashlib import sha256
from pathlib import Path


def sha256_file(file_path: Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""
    digest = sha256()
    with file_path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()
