"""Content hashing and canonical serialization.

These are the primitives behind dataset manifests and version snapshots:
a dataset's identity is the SHA-256 of its canonically-serialized records,
so identical content always hashes identically regardless of how it was
produced or in what order dict keys were written.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def canonical_json(obj: Any, *, default: Callable[[Any], str] | None = None) -> str:
    """Serialize to JSON with sorted keys and no incidental whitespace.

    ``default`` handles non-JSON scalars (e.g. timestamps from Arrow rows);
    pass ``str`` to stringify them deterministically.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=default)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_texts(texts: Iterable[str]) -> str:
    """Order-sensitive digest of a sequence of strings (one line each)."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
