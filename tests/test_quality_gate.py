import json
from pathlib import Path

import pytest

from crucible.assay import score_gate
from crucible.ingest import JsonlSource, land
from crucible.quality import QualityConfig, run_gate, write_report
from crucible.storage import Catalog, Layer
from crucible.synth import SynthConfig, generate_corpus, write_jsonl

# Same corpus the smoke test uses; all expectations below are measured.
SYNTH = SynthConfig(seed=42, n_docs=400)


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(generate_corpus(SYNTH), corpus)
    cat = Catalog(tmp_path / "catalog")
    land(JsonlSource(corpus, batch_size=97), cat, "synth", "test")
    return cat


def _quarantined_ids(catalog: Catalog) -> set[str]:
    if not catalog.parts(Layer.QUARANTINE, "synth"):
        return set()
    return set(catalog.read(Layer.QUARANTINE, "synth").column("id").to_pylist())


def test_gate_promotes_and_quarantines_with_perfect_measured_scores(catalog: Catalog) -> None:
    result = run_gate(catalog, "synth", QualityConfig())
    assert result.verdict == "promoted"
    assert result.input_rows == 400
    assert result.promoted_rows + result.quarantined_rows == 400
    assert result.quarantined_rows == 40  # 24 junk + 16 pii, measured

    score = score_gate(catalog.read(Layer.BRONZE, "synth"), _quarantined_ids(catalog))
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.recall_by_kind == {
        "junk_boilerplate": 1.0,
        "junk_empty": 1.0,
        "junk_mojibake": 1.0,
        "junk_short": 1.0,
        "pii": 1.0,
    }


def test_repeated_sentences_rule_trades_precision_for_nothing_here(catalog: Catalog) -> None:
    """The measured reason the rule is opt-in: +0 recall, -precision."""
    cfg = QualityConfig(
        rules=[
            "non_empty",
            "min_words",
            "no_mojibake",
            "no_boilerplate_markers",
            "no_repeated_sentences",
            "no_pii",
        ]
    )
    run_gate(catalog, "synth", cfg)
    score = score_gate(catalog.read(Layer.BRONZE, "synth"), _quarantined_ids(catalog))
    assert score.recall == 1.0
    assert score.precision < 1.0
    assert score.false_positive_kinds.get("clean", 0) > 0


def test_quarantine_carries_reject_reasons(catalog: Catalog) -> None:
    run_gate(catalog, "synth", QualityConfig())
    quarantine = catalog.read(Layer.QUARANTINE, "synth")
    assert "reject_reasons" in quarantine.column_names
    reasons = set()
    for value in quarantine.column("reject_reasons").to_pylist():
        reasons.update(value.split("|"))
    assert "no_pii" in reasons
    assert "non_empty" in reasons


def test_gate_blocks_when_reject_rate_exceeds_ceiling(catalog: Catalog) -> None:
    result = run_gate(catalog, "synth", QualityConfig(max_reject_rate=0.01))
    assert result.verdict == "blocked"
    assert result.promoted_rows == 0
    assert catalog.parts(Layer.SILVER, "synth") == []  # nothing leaked
    assert any("exceeds max" in note for note in result.notes)


def test_gate_rerun_is_deterministic(catalog: Catalog) -> None:
    run_gate(catalog, "synth", QualityConfig())
    first = [p.name for p in catalog.parts(Layer.SILVER, "synth")]
    run_gate(catalog, "synth", QualityConfig())
    assert [p.name for p in catalog.parts(Layer.SILVER, "synth")] == first


def test_gate_preserves_gt_columns_without_reading_them(catalog: Catalog) -> None:
    run_gate(catalog, "synth", QualityConfig())
    silver = catalog.read(Layer.SILVER, "synth")
    assert {"gt_kind", "gt_dup_of"} <= set(silver.column_names)


def test_gate_requires_text_column(tmp_path: Path) -> None:
    import pyarrow as pa

    cat = Catalog(tmp_path / "catalog")
    cat.write_part(pa.table({"id": ["a"]}), Layer.BRONZE, "notext", "part-0001")
    with pytest.raises(ValueError, match="no 'text' column"):
        run_gate(cat, "notext", QualityConfig())


def test_report_written_as_json_and_markdown(catalog: Catalog, tmp_path: Path) -> None:
    cfg = QualityConfig()
    result = run_gate(catalog, "synth", cfg)
    json_path, md_path = write_report(result, cfg, tmp_path / "out")
    payload = json.loads(json_path.read_text())
    assert payload["verdict"] == "promoted"
    assert payload["config"]["min_words"] == cfg.min_words
    md = md_path.read_text()
    assert "Verdict: promoted" in md
    assert "`no_pii`" in md
