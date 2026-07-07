"""From-scratch MinHash + banded LSH (Broder 1997; MMDS ch. 3).

Construction: documents are shingled into word n-grams, each shingle hashed
to a 64-bit integer (blake2b — NOT Python's ``hash()``, whose per-process
randomization would break determinism). A MinHash signature applies
``num_perm`` universal hash functions ``(a*x + b) mod p`` (p = 2^61 - 1)
and keeps the minimum per function; the probability two signatures agree at
one position equals the Jaccard similarity of the shingle sets.

Banded LSH splits signatures into ``bands`` bands of ``rows`` rows; two
documents become a candidate pair iff some band matches exactly, giving the
familiar S-curve with inflection near ``(1/bands)^(1/rows)``. Candidates
are then verified with *exact* Jaccard over the shingle sets, so LSH tuning
affects recall and speed but never lets a false positive through
unverified.

Everything is deterministic given the seed: permutation coefficients come
from a seeded RNG stream, consistent with the project-wide convention.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable

import numpy as np

_MERSENNE_PRIME = np.uint64((1 << 61) - 1)


def shingles(text: str, size: int = 3) -> set[str]:
    """Word n-gram shingles; shorter texts collapse to one whole-text shingle."""
    words = text.split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _hash_shingle(shingle: str) -> np.uint64:
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, "big") & ((1 << 61) - 1))


class MinHasher:
    """Vectorized MinHash signatures over ``num_perm`` universal hashes."""

    def __init__(self, num_perm: int = 128, seed: int = 0) -> None:
        if num_perm < 2:
            raise ValueError("num_perm must be >= 2")
        self.num_perm = num_perm
        rng = random.Random(f"{seed}/minhash")
        prime = int(_MERSENNE_PRIME)
        self._a = np.array([rng.randrange(1, prime) for _ in range(num_perm)], dtype=np.uint64)
        self._b = np.array([rng.randrange(0, prime) for _ in range(num_perm)], dtype=np.uint64)

    def signature(self, doc_shingles: Iterable[str]) -> np.ndarray:
        hashes = np.array([_hash_shingle(s) for s in doc_shingles], dtype=np.uint64)
        if hashes.size == 0:
            return np.full(self.num_perm, _MERSENNE_PRIME, dtype=np.uint64)
        # (num_perm, n_shingles): (a*x + b) mod p, min over shingles.
        products = (self._a[:, None] * hashes[None, :] + self._b[:, None]) % _MERSENNE_PRIME
        return products.min(axis=1)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Fraction of agreeing signature positions ≈ Jaccard similarity."""
    return float(np.mean(sig_a == sig_b))


def lsh_candidate_pairs(signatures: list[np.ndarray], bands: int = 32) -> set[tuple[int, int]]:
    """Candidate pairs whose signatures collide in at least one band."""
    if not signatures:
        return set()
    num_perm = len(signatures[0])
    if num_perm % bands != 0:
        raise ValueError(f"bands ({bands}) must divide num_perm ({num_perm})")
    rows = num_perm // bands

    candidates: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = {}
        start = band * rows
        for doc_index, signature in enumerate(signatures):
            key = signature[start : start + rows].tobytes()
            buckets.setdefault(key, []).append(doc_index)
        for members in buckets.values():
            if len(members) > 1:
                for i, first in enumerate(members):
                    for second in members[i + 1 :]:
                        candidates.add((first, second))
    return candidates
