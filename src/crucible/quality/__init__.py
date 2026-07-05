"""Quality: validation rules, the bronze→silver promotion gate, quarantine,
and drift detection.

The gate is a pure function of (bronze content, QualityConfig): records
failing row-level rules are quarantined with their reasons; the rest are
promoted to silver. If the reject rate exceeds the configured ceiling the
whole promotion is blocked — a systematically broken source should never
leak a "mostly fine" silver dataset.

Pipeline code here never reads the synthetic ground-truth columns; measured
precision/recall against them lives in :mod:`crucible.assay`.
"""

from crucible.quality.drift import DatasetProfile, drift_report, population_stability_index
from crucible.quality.gate import GateResult, QualityConfig, run_gate
from crucible.quality.report import render_markdown, write_report
from crucible.quality.rules import RULES, evaluate_text

__all__ = [
    "RULES",
    "DatasetProfile",
    "GateResult",
    "QualityConfig",
    "drift_report",
    "evaluate_text",
    "population_stability_index",
    "render_markdown",
    "run_gate",
    "write_report",
]
