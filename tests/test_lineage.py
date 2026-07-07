from pathlib import Path

from crucible.dedup import DedupConfig, run_dedup
from crucible.ingest import JsonlSource, land
from crucible.lineage import LineageGraph, dataset_ref, emit_event
from crucible.quality import QualityConfig, run_gate
from crucible.storage import Catalog
from crucible.synth import SynthConfig, generate_corpus, write_jsonl


def _run_pipeline(tmp_path: Path) -> Catalog:
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(generate_corpus(SynthConfig(seed=11, n_docs=120)), corpus)
    catalog = Catalog(tmp_path / "catalog")
    land(JsonlSource(corpus, batch_size=50), catalog, "synth", "test")
    run_gate(catalog, "synth", QualityConfig())
    run_dedup(catalog, "synth", DedupConfig())
    return catalog


def test_pipeline_emits_full_lineage(tmp_path: Path) -> None:
    catalog = _run_pipeline(tmp_path)
    graph = LineageGraph.from_root(catalog.root)
    assert {"ingest:synth", "promote:synth", "dedup:synth"} <= set(graph.jobs)
    assert {"bronze/synth", "silver/synth", "quarantine/synth"} <= graph.datasets
    edges = graph.edges()
    assert ("bronze/synth", "job:promote:synth") in edges
    assert ("job:promote:synth", "silver/synth") in edges
    assert ("job:promote:synth", "quarantine/synth") in edges
    assert ("silver/synth", "job:dedup:synth") in edges


def test_upstream_traces_back_to_bronze_and_source(tmp_path: Path) -> None:
    catalog = _run_pipeline(tmp_path)
    graph = LineageGraph.from_root(catalog.root)
    upstream = graph.upstream("silver/synth")
    assert "bronze/synth" in upstream
    assert "test" in upstream  # the external source_name passed to land()


def test_rerun_supersedes_job_event(tmp_path: Path) -> None:
    catalog = _run_pipeline(tmp_path)
    run_gate(catalog, "synth", QualityConfig())  # re-run promote
    graph = LineageGraph.from_root(catalog.root)
    # Still one promote job (latest event wins), not an accumulation.
    assert sum(1 for name in graph.jobs if name.startswith("promote")) == 1


def test_mermaid_render_contains_nodes_and_edges(tmp_path: Path) -> None:
    catalog = _run_pipeline(tmp_path)
    mermaid = LineageGraph.from_root(catalog.root).to_mermaid()
    assert mermaid.startswith("flowchart LR")
    assert 'bronze_synth["bronze/synth"]' in mermaid
    assert "job_promote_synth" in mermaid
    assert "bronze_synth --> job_promote_synth" in mermaid


def test_blocked_promotion_emits_event_without_outputs(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(generate_corpus(SynthConfig(seed=11, n_docs=120)), corpus)
    catalog = Catalog(tmp_path / "catalog")
    land(JsonlSource(corpus, batch_size=50), catalog, "synth", "test")
    run_gate(catalog, "synth", QualityConfig(max_reject_rate=0.0001))
    graph = LineageGraph.from_root(catalog.root)
    event = graph.jobs["promote:synth"]
    assert event["outputs"] == []
    assert event["run"]["facets"]["verdict"] == "blocked"


def test_empty_root_yields_empty_graph(tmp_path: Path) -> None:
    graph = LineageGraph.from_root(tmp_path)
    assert graph.jobs == {}
    assert graph.edges() == []
    assert graph.upstream("anything") == set()


def test_manual_emit_round_trip(tmp_path: Path) -> None:
    emit_event(
        tmp_path,
        job="custom:step",
        inputs=[dataset_ref("bronze/a", "hash1", 10)],
        outputs=[dataset_ref("gold/b", "hash2", 5)],
    )
    graph = LineageGraph.from_root(tmp_path)
    assert graph.upstream("gold/b") == {"bronze/a"}
