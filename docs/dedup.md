# Deduplication

Dedup runs after the quality gate and narrows the silver contract to
"validated **and deduplicated**". Two passes share one union-find:

1. **Exact**: records grouped by SHA-256 of normalized text
   (`none` | `whitespace` | `aggressive` normalization).
2. **Near-duplicate**: word 3-gram shingles → MinHash signatures (128
   universal hashes `(a·x + b) mod 2^61−1`, seed-deterministic, vectorized
   with numpy) → banded LSH (32 bands × 4 rows, S-curve inflection ≈ 0.42)
   → candidate pairs verified with **exact** Jaccard before clustering, so
   LSH tuning affects recall and speed but never admits an unverified
   false positive.

Within each cluster the record with the smallest id survives (monotone in
event time for the synthetic corpus). Survivor choice follows the id key,
not row position — Parquet parts are content-hash-named, so read order is
arbitrary; measurement caught the naive keep-first silently keeping copies
before this was fixed.

References: Broder (1997), *On the resemblance and containment of
documents*; Leskovec, Rajaraman & Ullman, *Mining of Massive Datasets*,
ch. 3; Lee et al. (2022), *Deduplicating Training Data Makes Language
Models Better* (arXiv:2107.06499), the motivation for the Phase 8 dedup
ablation.

## Backends

`backend: native` (default, zero extra dependencies) or `backend:
datasketch` (`.[dedup]` extra). The datasketch adapter is forced onto the
same explicit LSH banding — its threshold-based auto-tuning would otherwise
pick a more conservative S-curve and silently lose candidate recall (found
by the backend-equivalence test). Both backends are tested; in dev installs
datasketch is present so the equivalence test really runs.

## Measured results (seed-42 corpus, 360 pre-dedup silver rows)

Reproduce: `make smoke`, or
`crucible score-dedup --dataset synth --sweep 0.3,0.4,0.5,0.6,0.7`.

| Threshold | Removed | Precision | Recall | F1 | exact recall | near recall |
|---|---|---|---|---|---|---|
| 0.3 | 249 | 0.22 | 0.88 | 0.36 | 1.0 | 0.80 |
| 0.4 | 170 | 0.33 | 0.88 | 0.48 | 1.0 | 0.80 |
| **0.5 (default)** | **67** | **0.78** | **0.81** | **0.79** | **1.0** | **0.70** |
| 0.6 | 34 | 1.00 | 0.53 | 0.69 | 1.0 | 0.25 |
| 0.7 | 24 | 1.00 | 0.38 | 0.55 | 1.0 | 0.00 |

Reading the table honestly:

- **Exact-dup recall is 1.0 everywhere** — the hash pass is
  threshold-independent.
- **Near-dup recall tops out at 0.8**: planted near-dups carry 8–15% token
  edits, which puts their shingle-Jaccard band deliberately *straddling*
  usable thresholds. Perfect near-dup recall is not achievable here without
  destroying precision — by design, so the Phase 8 ablation has a real
  tradeoff to study.
- **Below 0.5 precision collapses** because clean template-generated docs
  share enough phrasing to cross J ≥ 0.4 — the synthetic analogue of
  boilerplate-heavy web text.
- One "false positive" at 0.5 is a **byte-identical accidental template
  collision labeled clean** — ground-truth incompleteness, counted
  separately as `fp_unlabeled_exact` rather than silently forgiven.
