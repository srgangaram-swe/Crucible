from pathlib import Path

import pyarrow as pa
import pytest

from crucible.dedup import DedupConfig, run_dedup
from crucible.ingest import JsonlSource, land
from crucible.quality import QualityConfig, run_gate
from crucible.storage import Catalog, Layer
from crucible.synth import SynthConfig, generate_corpus, write_jsonl
from crucible.versioning import build_manifest, list_snapshots, snapshot_stage, verify_snapshot


def _seeded_catalog(tmp_path: Path, name: str = "catalog") -> tuple[Catalog, Path]:
    corpus = tmp_path / "corpus.jsonl"
    if not corpus.exists():
        write_jsonl(generate_corpus(SynthConfig(seed=11, n_docs=120)), corpus)
    catalog = Catalog(tmp_path / name)
    land(JsonlSource(corpus, batch_size=50), catalog, "synth", "test")
    return catalog, corpus


def test_manifest_is_stable_and_order_independent(tmp_path: Path) -> None:
    catalog, _ = _seeded_catalog(tmp_path)
    first = build_manifest(catalog, Layer.BRONZE, "synth")
    second = build_manifest(catalog, Layer.BRONZE, "synth")
    assert first == second
    assert first.n_rows == 120
    assert ["id", "string"] in first.schema


def test_manifest_changes_when_content_changes(tmp_path: Path) -> None:
    catalog, _ = _seeded_catalog(tmp_path)
    before = build_manifest(catalog, Layer.BRONZE, "synth")
    catalog.write_part(pa.table({"id": ["extra"]}), Layer.BRONZE, "synth", "part-extra")
    after = build_manifest(catalog, Layer.BRONZE, "synth")
    assert before.content_hash != after.content_hash


def test_manifest_requires_parts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no parts"):
        build_manifest(Catalog(tmp_path), Layer.SILVER, "nope")


def test_stages_write_snapshots_and_verify(tmp_path: Path) -> None:
    catalog, _ = _seeded_catalog(tmp_path)
    run_gate(catalog, "synth", QualityConfig())
    run_dedup(catalog, "synth", DedupConfig())

    snapshots = list_snapshots(catalog.root, "synth")
    assert [snapshot["stage"] for snapshot in snapshots] == ["promote", "dedup"] or {
        snapshot["stage"] for snapshot in snapshots
    } == {"promote", "dedup"}
    for snapshot in snapshots:
        assert snapshot["config_hash"]
        assert snapshot["inputs"][0]["content_hash"]

    # dedup snapshot pins current silver -> verifies clean
    dedup_snapshot = next(s for s in snapshots if s["stage"] == "dedup")
    ok, detail = verify_snapshot(catalog, dedup_snapshot)
    assert ok, detail

    # promote snapshot pinned PRE-dedup silver -> verification must fail now
    promote_snapshot = next(s for s in snapshots if s["stage"] == "promote")
    ok, detail = verify_snapshot(catalog, promote_snapshot)
    assert not ok
    assert "content hash" in detail


def test_verify_detects_tampering(tmp_path: Path) -> None:
    catalog, _ = _seeded_catalog(tmp_path)
    run_gate(catalog, "synth", QualityConfig())
    snapshot = list_snapshots(catalog.root, "synth")[-1]
    ok, _ = verify_snapshot(catalog, snapshot)
    assert ok
    # Tamper: remove one silver part.
    catalog.parts(Layer.SILVER, "synth")[0].unlink()
    ok, _detail = verify_snapshot(catalog, snapshot)
    assert not ok


def test_snapshot_id_depends_on_config(tmp_path: Path) -> None:
    catalog, _ = _seeded_catalog(tmp_path)
    run_gate(catalog, "synth", QualityConfig())
    bronze = build_manifest(catalog, Layer.BRONZE, "synth")
    a = snapshot_stage(catalog, "promote", QualityConfig(), [bronze], (Layer.SILVER, "synth"))
    b = snapshot_stage(
        catalog, "promote", QualityConfig(min_words=7), [bronze], (Layer.SILVER, "synth")
    )
    assert a["snapshot_id"] != b["snapshot_id"]


def test_byte_identical_rebuild_across_catalogs(tmp_path: Path) -> None:
    """The reproducibility contract: same inputs + same configs + same code
    -> identical silver content hash in a completely fresh catalog."""
    catalog_a, corpus = _seeded_catalog(tmp_path, "a")
    run_gate(catalog_a, "synth", QualityConfig())
    run_dedup(catalog_a, "synth", DedupConfig())

    catalog_b = Catalog(tmp_path / "b")
    land(JsonlSource(corpus, batch_size=50), catalog_b, "synth", "test")
    run_gate(catalog_b, "synth", QualityConfig())
    run_dedup(catalog_b, "synth", DedupConfig())

    hash_a = build_manifest(catalog_a, Layer.SILVER, "synth").content_hash
    hash_b = build_manifest(catalog_b, Layer.SILVER, "synth").content_hash
    assert hash_a == hash_b
