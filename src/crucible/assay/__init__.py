"""Assay: measurement of pipeline-stage quality against planted ground truth.

An assay is the test of a metal's purity — this package is the *only* place
allowed to read the synthetic corpus's ``gt_*`` columns. Pipeline stages
rediscover defects blind; assay scores how well they did, which is what
makes reported precision/recall real measurements instead of claims.

Grows into the full experiment harness in Phase 8.
"""

from crucible.assay.scoring import (
    DedupScore,
    GateScore,
    score_dedup,
    score_gate,
    sweep_dedup_thresholds,
)

__all__ = ["DedupScore", "GateScore", "score_dedup", "score_gate", "sweep_dedup_thresholds"]
