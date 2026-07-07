"""Score the quality gate against the synthetic corpus's planted defects.

Definitions: a record is a true *defect* iff its ``gt_kind`` is a junk
variant or ``pii``. Duplicates (exact/near) are deliberately counted as
non-defects here — removing them is the dedup stage's job, and a gate that
quarantined them would be over-reaching (and scored down via precision).

- TP: defect, quarantined      FP: non-defect, quarantined
- FN: defect, promoted         TN: non-defect, promoted
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pyarrow as pa

_DEFECT_PREFIX = "junk_"
_DEFECT_KINDS = ("pii",)


class GroundTruthUnavailable(Exception):
    """The dataset lacks gt_* columns; scoring only works on synthetic data."""


def _is_defect(gt_kind: str) -> bool:
    return gt_kind.startswith(_DEFECT_PREFIX) or gt_kind in _DEFECT_KINDS


@dataclass(frozen=True, slots=True)
class GateScore:
    n_records: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    recall_by_kind: dict[str, float]
    false_positive_kinds: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_gate(bronze: pa.Table, quarantined_ids: set[str]) -> GateScore:
    """Exact precision/recall/F1 of quarantine decisions vs ground truth."""
    if "gt_kind" not in bronze.column_names or "id" not in bronze.column_names:
        raise GroundTruthUnavailable(
            "scoring requires id and gt_kind columns (synthetic corpora only)"
        )

    ids = bronze.column("id").to_pylist()
    kinds = bronze.column("gt_kind").to_pylist()

    tp = fp = fn = tn = 0
    kind_totals: dict[str, int] = {}
    kind_caught: dict[str, int] = {}
    fp_kinds: dict[str, int] = {}
    for record_id, kind in zip(ids, kinds, strict=True):
        quarantined = record_id in quarantined_ids
        if _is_defect(kind):
            kind_totals[kind] = kind_totals.get(kind, 0) + 1
            if quarantined:
                tp += 1
                kind_caught[kind] = kind_caught.get(kind, 0) + 1
            else:
                fn += 1
        elif quarantined:
            fp += 1
            fp_kinds[kind] = fp_kinds.get(kind, 0) + 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return GateScore(
        n_records=len(ids),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        recall_by_kind={
            kind: round(kind_caught.get(kind, 0) / total, 4)
            for kind, total in sorted(kind_totals.items())
        },
        false_positive_kinds=dict(sorted(fp_kinds.items())),
    )
