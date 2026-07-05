import json
from pathlib import Path

import pytest

from crucible.synth import (
    RecordKind,
    SynthConfig,
    corpus_sha256,
    generate_corpus,
    generation_report,
    write_jsonl,
    write_parquet,
)

CFG = SynthConfig(seed=1, n_docs=300)


def test_determinism_same_seed() -> None:
    assert corpus_sha256(generate_corpus(CFG)) == corpus_sha256(generate_corpus(CFG))


def test_different_seeds_differ() -> None:
    other = SynthConfig(seed=2, n_docs=300)
    assert corpus_sha256(generate_corpus(CFG)) != corpus_sha256(generate_corpus(other))


def test_counts_match_configured_rates() -> None:
    records = generate_corpus(CFG)
    assert len(records) == CFG.n_docs
    by_kind: dict[str, int] = {}
    for record in records:
        by_kind[record.gt_kind] = by_kind.get(record.gt_kind, 0) + 1
    assert by_kind["exact_dup"] == round(CFG.n_docs * CFG.exact_dup_rate)
    assert by_kind["near_dup"] == round(CFG.n_docs * CFG.near_dup_rate)
    assert by_kind["pii"] == round(CFG.n_docs * CFG.pii_rate)
    junk = sum(count for kind, count in by_kind.items() if kind.startswith("junk_"))
    assert junk == round(CFG.n_docs * CFG.junk_rate)


def test_duplicate_ground_truth_is_consistent() -> None:
    records = generate_corpus(CFG)
    by_id = {record.id: record for record in records}
    for record in records:
        if record.gt_kind == "exact_dup":
            assert record.gt_dup_of is not None
            assert record.text == by_id[record.gt_dup_of].text
        elif record.gt_kind == "near_dup":
            assert record.gt_dup_of is not None
            original = by_id[record.gt_dup_of]
            assert record.text != original.text
            # Near-dups must stay lexically close to their original.
            a, b = set(record.text.split()), set(original.text.split())
            jaccard = len(a & b) / len(a | b)
            assert jaccard > 0.5, f"near-dup drifted too far (jaccard={jaccard:.2f})"
        else:
            assert record.gt_dup_of is None


def test_records_are_time_ordered_with_monotone_ids() -> None:
    records = generate_corpus(CFG)
    timestamps = [record.timestamp for record in records]
    assert timestamps == sorted(timestamps)
    assert [record.id for record in records] == [f"rec-{i:06d}" for i in range(len(records))]


def test_sources_come_from_config() -> None:
    records = generate_corpus(CFG)
    assert {record.source for record in records} <= set(CFG.domain_weights)


def test_config_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="unknown domains"):
        SynthConfig(domain_weights={"astrology": 1.0})


def test_config_rejects_excessive_defect_rates() -> None:
    with pytest.raises(ValueError, match="defect rates"):
        SynthConfig(exact_dup_rate=0.3, near_dup_rate=0.3, junk_rate=0.3, pii_rate=0.3)


def test_jsonl_round_trip(tmp_path: Path) -> None:
    records = generate_corpus(SynthConfig(seed=3, n_docs=50))
    path = tmp_path / "corpus.jsonl"
    write_jsonl(records, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 50
    first = json.loads(lines[0])
    assert set(first) == {"id", "text", "source", "timestamp", "gt_kind", "gt_dup_of"}


def test_parquet_round_trip(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    records = generate_corpus(SynthConfig(seed=3, n_docs=50))
    path = tmp_path / "corpus.parquet"
    write_parquet(records, path)
    table = pq.read_table(path)
    assert table.num_rows == 50
    assert table.column("id").to_pylist() == [record.id for record in records]


def test_generation_report_shape() -> None:
    records = generate_corpus(CFG)
    report = generation_report(CFG, records)
    assert report["n_records"] == CFG.n_docs
    assert report["corpus_sha256"] == corpus_sha256(records)
    assert isinstance(report["by_kind"], dict)


def test_record_kind_junk_predicate() -> None:
    assert RecordKind.JUNK_EMPTY.is_junk
    assert not RecordKind.CLEAN.is_junk
