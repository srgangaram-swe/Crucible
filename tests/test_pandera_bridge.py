"""The pandera bridge is a second, independent opinion on gate output:
silver must satisfy the declarative contract that mirrors the native rules."""

from pathlib import Path

import pyarrow as pa
import pytest

pandera = pytest.importorskip("pandera")

from crucible.ingest import JsonlSource, land  # noqa: E402
from crucible.quality import QualityConfig, run_gate  # noqa: E402
from crucible.quality.pandera_bridge import corpus_schema, validate_table  # noqa: E402
from crucible.storage import Catalog, Layer  # noqa: E402
from crucible.synth import SynthConfig, generate_corpus, write_jsonl  # noqa: E402


def test_silver_output_satisfies_declarative_contract(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(generate_corpus(SynthConfig(seed=42, n_docs=200)), corpus)
    catalog = Catalog(tmp_path / "catalog")
    land(JsonlSource(corpus, batch_size=97), catalog, "synth", "test")
    cfg = QualityConfig()
    run_gate(catalog, "synth", cfg)

    silver = catalog.read(Layer.SILVER, "synth")
    validate_table(silver, min_words=cfg.min_words)  # must not raise


def test_contract_rejects_duplicate_ids() -> None:
    bad = pa.table(
        {
            "id": ["a", "a"],
            "text": ["five words are right here", "five words are right here"],
            "source": ["news", "news"],
            "timestamp": ["2026-01-01T00:00:00+00:00"] * 2,
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        validate_table(bad)


def test_contract_rejects_short_text() -> None:
    bad = pa.table(
        {
            "id": ["a"],
            "text": ["nope"],
            "source": ["news"],
            "timestamp": ["2026-01-01T00:00:00+00:00"],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        validate_table(bad, min_words=5)


def test_schema_allows_gt_passthrough_columns() -> None:
    schema = corpus_schema()
    assert schema.strict is False
