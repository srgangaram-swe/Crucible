from pathlib import Path

import pytest

from crucible.ingest import JsonlSource, land
from crucible.lineage import LineageGraph
from crucible.quality import QualityConfig, run_gate
from crucible.shards import ByteTokenizer, ShardConfig, ShardReader, build_shards
from crucible.storage import Catalog, Layer
from crucible.synth import SynthConfig, generate_corpus, write_jsonl
from crucible.versioning import build_manifest, list_snapshots

CFG = ShardConfig(seq_len=64, sequences_per_shard=50, seed=0)


def _silver_catalog(base: Path, name: str = "catalog") -> Catalog:
    corpus = base / "corpus.jsonl"
    if not corpus.exists():
        write_jsonl(generate_corpus(SynthConfig(seed=13, n_docs=150)), corpus)
    catalog = Catalog(base / name)
    land(JsonlSource(corpus, batch_size=60), catalog, "synth", "test")
    run_gate(catalog, "synth", QualityConfig())
    return catalog


@pytest.fixture(scope="module")
def sharded(tmp_path_factory: pytest.TempPathFactory) -> tuple[Catalog, object]:
    base = tmp_path_factory.mktemp("shards")
    catalog = _silver_catalog(base)
    result = build_shards(catalog, "synth", CFG)
    return catalog, result


def test_tokenizer_round_trip() -> None:
    tokenizer = ByteTokenizer()
    for text in ["hello world", "café — résumé", ""]:
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode([ByteTokenizer.EOS, 104, 105, ByteTokenizer.PAD]) == "hi"
    assert ByteTokenizer.vocab_size == 259


def test_packing_shapes_and_counts(sharded: tuple[Catalog, object]) -> None:
    catalog, result = sharded
    assert result.n_sequences > 0  # type: ignore[attr-defined]
    assert result.n_tokens == result.n_sequences * (CFG.seq_len + 1)  # type: ignore[attr-defined]
    assert result.dropped_tail_tokens <= CFG.seq_len  # type: ignore[attr-defined]
    table = catalog.read(Layer.GOLD, "synth_shards")
    lengths = {len(tokens) for tokens in table.column("tokens").to_pylist()}
    assert lengths == {CFG.seq_len + 1}
    assert table.num_rows == result.n_sequences  # type: ignore[attr-defined]


def test_shards_are_byte_identical_across_rebuilds(tmp_path: Path) -> None:
    catalog_a = _silver_catalog(tmp_path, "a")
    catalog_b = _silver_catalog(tmp_path, "b")
    build_shards(catalog_a, "synth", CFG)
    build_shards(catalog_b, "synth", CFG)
    hash_a = build_manifest(catalog_a, Layer.GOLD, "synth_shards").content_hash
    hash_b = build_manifest(catalog_b, Layer.GOLD, "synth_shards").content_hash
    assert hash_a == hash_b
    # Different seed -> different document order -> different content.
    build_shards(catalog_b, "synth", ShardConfig(seq_len=64, seed=1))
    assert build_manifest(catalog_b, Layer.GOLD, "synth_shards").content_hash != hash_a


def test_shards_record_lineage_and_snapshot(sharded: tuple[Catalog, object]) -> None:
    catalog, _ = sharded
    graph = LineageGraph.from_root(catalog.root)
    assert "shard:synth" in graph.jobs
    assert "silver/synth" in graph.upstream("gold/synth_shards")
    stages = {snapshot["stage"] for snapshot in list_snapshots(catalog.root, "synth_shards")}
    assert "shard" in stages


def test_gold_shards_queryable_via_duckdb(sharded: tuple[Catalog, object]) -> None:
    catalog, result = sharded
    rows = catalog.query("SELECT count(*) AS n FROM gold_synth_shards")
    assert rows == [{"n": result.n_sequences}]  # type: ignore[attr-defined]


def test_reader_visits_every_sequence_exactly_once(sharded: tuple[Catalog, object]) -> None:
    catalog, result = sharded
    reader = ShardReader(catalog, "synth_shards", seed=0, shuffle_buffer=16)
    seen = list(reader.iterate(epoch=0))
    assert len(seen) == result.n_sequences  # type: ignore[attr-defined]
    assert reader.n_sequences() == result.n_sequences  # type: ignore[attr-defined]


def test_reader_is_deterministic_and_shuffles(sharded: tuple[Catalog, object]) -> None:
    catalog, _ = sharded
    reader = ShardReader(catalog, "synth_shards", seed=0, shuffle_buffer=16)
    first = list(reader.iterate(epoch=0))
    second = list(reader.iterate(epoch=0))
    assert first == second  # same seed + epoch -> identical order
    other_epoch = list(reader.iterate(epoch=1))
    assert other_epoch != first  # epochs reshuffle
    unshuffled = list(ShardReader(catalog, "synth_shards", shuffle_buffer=1).iterate(0))
    assert first != unshuffled  # the buffer actually permutes


def test_resume_mid_epoch_is_exact(sharded: tuple[Catalog, object]) -> None:
    """Crash-resume: full run == head + resumed tail, element for element."""
    catalog, result = sharded
    reader = ShardReader(catalog, "synth_shards", seed=0, shuffle_buffer=16)
    full = list(reader.iterate(epoch=0))

    interrupted = reader.iterate(epoch=0)
    head = [next(interrupted) for _ in range(7)]
    state = interrupted.state()
    assert state == {"epoch": 0, "consumed": 7}

    tail = list(reader.iterate(epoch=0, resume_state=state))
    assert head + tail == full
    assert len(tail) == result.n_sequences - 7  # type: ignore[attr-defined]


def test_resume_rejects_wrong_epoch(sharded: tuple[Catalog, object]) -> None:
    catalog, _ = sharded
    reader = ShardReader(catalog, "synth_shards")
    with pytest.raises(ValueError, match="epoch"):
        reader.iterate(epoch=1, resume_state={"epoch": 0, "consumed": 3})


def test_build_rejects_too_small_corpus(tmp_path: Path) -> None:
    catalog = _silver_catalog(tmp_path)
    with pytest.raises(ValueError, match="too few tokens"):
        build_shards(catalog, "synth", ShardConfig(seq_len=10_000_000))
