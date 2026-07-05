import pyarrow as pa
import pytest

from crucible.assay import score_gate
from crucible.assay.scoring import GroundTruthUnavailable


def _table(rows: list[tuple[str, str]]) -> pa.Table:
    return pa.table(
        {
            "id": [record_id for record_id, _ in rows],
            "gt_kind": [kind for _, kind in rows],
        }
    )


def test_exact_confusion_matrix() -> None:
    table = _table(
        [
            ("a", "junk_empty"),  # quarantined -> TP
            ("b", "pii"),  # quarantined -> TP
            ("c", "junk_short"),  # promoted    -> FN
            ("d", "clean"),  # quarantined -> FP
            ("e", "clean"),  # promoted    -> TN
            ("f", "exact_dup"),  # promoted    -> TN (dups are not defects)
        ]
    )
    score = score_gate(table, quarantined_ids={"a", "b", "d"})
    assert (score.tp, score.fp, score.fn, score.tn) == (2, 1, 1, 2)
    assert score.precision == pytest.approx(2 / 3, abs=1e-4)
    assert score.recall == pytest.approx(2 / 3, abs=1e-4)
    assert score.recall_by_kind == {"junk_empty": 1.0, "junk_short": 0.0, "pii": 1.0}
    assert score.false_positive_kinds == {"clean": 1}


def test_duplicates_count_against_precision_when_quarantined() -> None:
    table = _table([("a", "near_dup"), ("b", "junk_empty")])
    score = score_gate(table, quarantined_ids={"a", "b"})
    assert score.fp == 1
    assert score.false_positive_kinds == {"near_dup": 1}


def test_empty_quarantine_on_defect_free_data_is_perfect() -> None:
    table = _table([("a", "clean"), ("b", "clean")])
    score = score_gate(table, quarantined_ids=set())
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.recall_by_kind == {}


def test_requires_ground_truth_columns() -> None:
    with pytest.raises(GroundTruthUnavailable):
        score_gate(pa.table({"id": ["a"], "text": ["t"]}), set())
