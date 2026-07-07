"""Deduplication: exact (content hash) and near-duplicate (MinHash + LSH).

Runs after the quality gate and narrows the silver contract to "validated
AND deduplicated". Exact dedup groups records by normalized-text hash;
near-dup detection shingles text into word n-grams, sketches them with
MinHash, finds candidate pairs via banded LSH, verifies candidates with
exact Jaccard similarity, and clusters with union-find. Within every
duplicate cluster the earliest record (smallest id = earliest event time)
is kept.

The default backend is a from-scratch, seed-deterministic implementation
with no dependencies beyond the core; ``backend: datasketch`` switches to
the datasketch library (``.[dedup]`` extra) behind the same interface.
Measured precision/recall against the planted ``gt_dup_of`` labels lives in
:mod:`crucible.assay` — dedup itself never reads ground truth.
"""

from crucible.dedup.exact import exact_duplicate_groups
from crucible.dedup.minhash import MinHasher, jaccard, lsh_candidate_pairs, shingles
from crucible.dedup.pipeline import DedupConfig, DedupResult, find_duplicates, run_dedup

__all__ = [
    "DedupConfig",
    "DedupResult",
    "MinHasher",
    "exact_duplicate_groups",
    "find_duplicates",
    "jaccard",
    "lsh_candidate_pairs",
    "run_dedup",
    "shingles",
]
