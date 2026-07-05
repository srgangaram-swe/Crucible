"""Catalog-level dedup with MEASURED expectations on the seed-42 corpus.

Every number asserted here was produced by running the pipeline; if a
change shifts them, the change altered dedup behavior and the new numbers
must be re-measured and re-justified, not fudged.
"""

from pathlib import Path

import pyarrow as pa
import pytest

from crucible.assay import score_dedup, sweep_dedup_thresholds
from crucible.assay.scoring import GroundTruthUnavailable
from crucible.dedup import DedupConfig, find_duplicates, run_dedup
from crucible.dedup.pipeline import write_dedup_report
from crucible.ingest import JsonlSource, land
from crucible.quality import QualityConfig, run_gate
from crucible.storage import Catalog, Layer
from crucible.synth import SynthConfig, generate_corpus, write_jsonl

SYNTH = SynthConfig(seed=42, n_docs=400)


@pytest.fixture(scope="module")
def gated_catalog(tmp_path_factory: pytest.TempPathFactory) -> Catalog:
    tmp_path = tmp_path_factory.mktemp("dedup")
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(generate_corpus(SYNTH), corpus)
    catalog = Catalog(tmp_path / "catalog")
    land(JsonlSource(corpus, batch_size=97), catalog, "synth", "test")
    run_gate(catalog, "synth", QualityConfig())
    return catalog


def test_default_dedup_measured_scores(gated_catalog: Catalog, tmp_path: Path) -> None:
    silver_pre = gated_catalog.read(Layer.SILVER, "synth")
    assert silver_pre.num_rows == 360

    result = run_dedup(gated_catalog, "synth", DedupConfig())
    assert result.kept_rows == 293
    assert result.removed_exact == 23
    assert result.removed_near == 44

    score = score_dedup(silver_pre, set(result.removed_ids))
    assert score.recall_by_kind["exact_dup"] == 1.0  # every planted exact copy removed
    assert score.recall_by_kind["near_dup"] == 0.7  # 12/40 mutated past J>=0.5
    assert score.precision >= 0.75
    assert score.f1 >= 0.75
    assert score.fp_unlabeled_exact == 1  # one accidental template collision

    # Re-promote so later tests see a fresh pre-dedup silver.
    run_gate(gated_catalog, "synth", QualityConfig())


def test_threshold_sweep_shape(gated_catalog: Catalog) -> None:
    silver_pre = gated_catalog.read(Layer.SILVER, "synth")
    sweep = sweep_dedup_thresholds(silver_pre, DedupConfig(), [0.4, 0.5, 0.6])
    by_threshold = {row["threshold"]: row for row in sweep}
    # Exact dups are threshold-independent (hash pass).
    assert all(row["recall_by_kind"]["exact_dup"] == 1.0 for row in sweep)
    # Lower threshold: more recall, worse precision; higher: the reverse.
    assert by_threshold[0.4]["recall"] > by_threshold[0.6]["recall"]
    assert by_threshold[0.6]["precision"] > by_threshold[0.4]["precision"]
    assert by_threshold[0.6]["precision"] == 1.0  # measured: zero FPs at 0.6


def test_datasketch_backend_agrees_on_planted_duplicates(gated_catalog: Catalog) -> None:
    pytest.importorskip("datasketch")
    silver_pre = gated_catalog.read(Layer.SILVER, "synth")
    sweep = sweep_dedup_thresholds(silver_pre, DedupConfig(backend="datasketch"), [0.5])
    row = sweep[0]
    assert row["recall_by_kind"]["exact_dup"] == 1.0
    assert row["recall_by_kind"]["near_dup"] >= 0.6  # different permutation family
    assert row["precision"] >= 0.7


def test_dedup_rerun_is_stable(gated_catalog: Catalog) -> None:
    run_gate(gated_catalog, "synth", QualityConfig())
    first = run_dedup(gated_catalog, "synth", DedupConfig())
    parts_first = [p.name for p in gated_catalog.parts(Layer.SILVER, "synth")]
    second = run_dedup(gated_catalog, "synth", DedupConfig())
    assert second.kept_rows == first.kept_rows
    assert second.removed_ids == []  # nothing left to remove
    assert [p.name for p in gated_catalog.parts(Layer.SILVER, "synth")] == parts_first
    run_gate(gated_catalog, "synth", QualityConfig())


def test_keep_first_survives_scrambled_row_order() -> None:
    """Representative choice must follow order_keys, not row position —
    Parquet part names are content hashes, so read order is arbitrary."""
    texts = ["same text here okay", "different entirely words", "same text here okay"]
    ids = ["rec-2", "rec-9", "rec-1"]  # the LAST row is the earliest record
    duplicates = find_duplicates(texts, DedupConfig(), order_keys=ids)
    assert duplicates.clusters == [[2, 0]]  # index 2 (rec-1) kept, index 0 removed
    assert duplicates.removed == [0]


def test_report_written(gated_catalog: Catalog, tmp_path: Path) -> None:
    run_gate(gated_catalog, "synth", QualityConfig())
    result = run_dedup(gated_catalog, "synth", DedupConfig())
    json_path, md_path = write_dedup_report(result, tmp_path)
    assert json_path.exists() and md_path.exists()
    assert "kept" in md_path.read_text()
    run_gate(gated_catalog, "synth", QualityConfig())


def test_score_dedup_requires_ground_truth() -> None:
    with pytest.raises(GroundTruthUnavailable):
        score_dedup(pa.table({"id": ["a"], "text": ["t"]}), set())


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="must divide"):
        DedupConfig(num_perm=128, bands=33)
    with pytest.raises(ValueError, match="unknown backend"):
        DedupConfig(backend="gpu")
