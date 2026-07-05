"""Score pipeline stages against the synthetic corpus's planted defects.

Gate scoring: a record is a true *defect* iff its ``gt_kind`` is a junk
variant or ``pii``. Duplicates (exact/near) are deliberately counted as
non-defects here — removing them is the dedup stage's job, and a gate that
quarantined them would be over-reaching (and scored down via precision).

Dedup scoring: a record is a true *duplicate* iff ``gt_dup_of`` is set (it
was planted as a copy of an earlier record). The keep-first policy means
the planted original should always survive; removing an original while
keeping its copy costs both precision and recall, which is the right
penalty.

- TP: positive, removed/quarantined    FP: negative, removed/quarantined
- FN: positive, kept/promoted          TN: negative, kept/promoted
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pyarrow as pa

from crucible.dedup.pipeline import DedupConfig, find_duplicates

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


@dataclass(frozen=True, slots=True)
class DedupScore:
    n_records: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    recall_by_kind: dict[str, float]  # exact_dup / near_dup
    false_positive_kinds: dict[str, int]
    # FPs that are byte-identical to another record yet labeled clean:
    # accidental template collisions the generator did not plan. These are
    # ground-truth incompleteness, not detector mistakes, and are reported
    # separately rather than silently forgiven.
    fp_unlabeled_exact: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_dedup(table: pa.Table, removed_ids: set[str]) -> DedupScore:
    """Exact precision/recall/F1 of removal decisions vs planted duplicates."""
    required = {"id", "gt_kind", "gt_dup_of"}
    if not required <= set(table.column_names):
        raise GroundTruthUnavailable(
            "scoring requires id, gt_kind, and gt_dup_of columns (synthetic corpora only)"
        )
    ids = table.column("id").to_pylist()
    kinds = table.column("gt_kind").to_pylist()
    dup_of = table.column("gt_dup_of").to_pylist()
    texts = table.column("text").to_pylist() if "text" in table.column_names else [None] * len(ids)
    text_counts: dict[str, int] = {}
    for text in texts:
        if text is not None:
            text_counts[text] = text_counts.get(text, 0) + 1

    tp = fp = fn = tn = 0
    fp_unlabeled_exact = 0
    kind_totals: dict[str, int] = {}
    kind_caught: dict[str, int] = {}
    fp_kinds: dict[str, int] = {}
    for record_id, kind, original, text in zip(ids, kinds, dup_of, texts, strict=True):
        removed = record_id in removed_ids
        if original is not None:
            kind_totals[kind] = kind_totals.get(kind, 0) + 1
            if removed:
                tp += 1
                kind_caught[kind] = kind_caught.get(kind, 0) + 1
            else:
                fn += 1
        elif removed:
            fp += 1
            fp_kinds[kind] = fp_kinds.get(kind, 0) + 1
            if text is not None and text_counts.get(text, 0) > 1:
                fp_unlabeled_exact += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DedupScore(
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
        fp_unlabeled_exact=fp_unlabeled_exact,
    )


def sweep_dedup_thresholds(
    table: pa.Table, cfg: DedupConfig, thresholds: list[float]
) -> list[dict[str, Any]]:
    """Re-run near-dup clustering at each threshold and score it.

    Run this on a *pre-dedup* table (e.g. silver right after promotion):
    after ``run_dedup`` the duplicates are gone and every sweep point would
    trivially score zero removals.
    """
    texts = [value or "" for value in table.column(cfg.text_column).to_pylist()]
    ids = [str(value) for value in table.column(cfg.id_column).to_pylist()]
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        point_cfg = cfg.model_copy(update={"threshold": threshold})
        duplicates = find_duplicates(texts, point_cfg, order_keys=ids)
        removed = {ids[index] for index in duplicates.removed}
        score = score_dedup(table, removed)
        rows.append(
            {
                "threshold": threshold,
                "removed": len(removed),
                "removed_exact": duplicates.removed_exact,
                "removed_near": duplicates.removed_near,
                "precision": score.precision,
                "recall": score.recall,
                "f1": score.f1,
                "recall_by_kind": score.recall_by_kind,
            }
        )
    return rows
