"""Assay: measurement of pipeline-stage quality against planted ground truth.

An assay is the test of a metal's purity — this package is the *only* place
allowed to read the synthetic corpus's ``gt_*`` columns. Pipeline stages
rediscover defects blind; assay scores how well they did, which is what
makes reported precision/recall real measurements instead of claims.

Grows into the full experiment harness in Phase 8.
"""

from crucible.assay.harness import ExperimentConfig, ExperimentResult, bootstrap_ci, run_experiment
from crucible.assay.scoring import (
    DedupScore,
    GateScore,
    score_dedup,
    score_gate,
    sweep_dedup_thresholds,
)
from crucible.assay.studies import STUDIES

__all__ = [
    "STUDIES",
    "DedupScore",
    "ExperimentConfig",
    "ExperimentResult",
    "GateScore",
    "bootstrap_ci",
    "run_experiment",
    "score_dedup",
    "score_gate",
    "sweep_dedup_thresholds",
]
