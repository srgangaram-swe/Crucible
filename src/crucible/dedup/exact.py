"""Exact duplicate detection via normalized-text content hashing."""

from __future__ import annotations

import re

from crucible.utils.hashing import sha256_text

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str, mode: str) -> str:
    """``none`` = byte-exact; ``whitespace`` folds runs of whitespace;
    ``aggressive`` also lowercases (catches trivially re-cased copies)."""
    if mode == "none":
        return text
    folded = _WHITESPACE.sub(" ", text).strip()
    if mode == "whitespace":
        return folded
    if mode == "aggressive":
        return folded.lower()
    raise ValueError(f"unknown normalize mode {mode!r}")


def exact_duplicate_groups(texts: list[str], mode: str = "whitespace") -> list[list[int]]:
    """Indices of byte/normalized-identical records, grouped; singletons
    omitted. Order within a group follows input order (earliest first)."""
    by_hash: dict[str, list[int]] = {}
    for index, text in enumerate(texts):
        by_hash.setdefault(sha256_text(normalize(text, mode)), []).append(index)
    return [group for group in by_hash.values() if len(group) > 1]
