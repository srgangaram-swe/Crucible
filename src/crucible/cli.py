"""The ``crucible`` command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from crucible import __version__
from crucible.assay import (
    STUDIES,
    ExperimentConfig,
    run_experiment,
    score_dedup,
    score_gate,
    sweep_dedup_thresholds,
)
from crucible.assay.scoring import GroundTruthUnavailable
from crucible.bench import run_bench
from crucible.config import load_config
from crucible.dedup import DedupConfig, run_dedup
from crucible.dedup.pipeline import write_dedup_report
from crucible.ingest import (
    IngestConfig,
    InMemoryBroker,
    StreamSource,
    land,
    open_source,
    replay_jsonl,
)
from crucible.lineage import LineageGraph
from crucible.observability import MetricsStore
from crucible.orchestrate import list_runs, run_pipeline
from crucible.quality import QualityConfig, drift_report, run_gate, write_report
from crucible.quality.drift import profile_table
from crucible.smoke import SmokeFailure, run_smoke
from crucible.storage import Catalog, Layer
from crucible.synth import (
    SynthConfig,
    generate_corpus,
    generation_report,
    write_jsonl,
    write_parquet,
)
from crucible.versioning import list_snapshots, verify_snapshot


@click.group()
@click.version_option(version=__version__, prog_name="crucible")
def main() -> None:
    """Crucible: a local-first data refinery for distributed training."""


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML config for the generator (see configs/synth_small.yaml).",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/raw/synth"),
    show_default=True,
    help="Output directory.",
)
@click.option(
    "--fmt",
    type=click.Choice(["jsonl", "parquet", "both"]),
    default="both",
    show_default=True,
)
@click.option("--seed", type=int, default=None, help="Override the config seed.")
@click.option("--n-docs", type=int, default=None, help="Override the config document count.")
def synth(
    config_path: Path | None,
    out_dir: Path,
    fmt: str,
    seed: int | None,
    n_docs: int | None,
) -> None:
    """Generate a deterministic synthetic corpus with ground-truth defect labels."""
    cfg = load_config(SynthConfig, config_path, seed=seed, n_docs=n_docs)
    records = generate_corpus(cfg)
    if fmt in ("jsonl", "both"):
        write_jsonl(records, out_dir / "corpus.jsonl")
    if fmt in ("parquet", "both"):
        write_parquet(records, out_dir / "corpus.parquet")
    report = generation_report(cfg, records)
    (out_dir / "generation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    click.echo(json.dumps(report, indent=2))


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML IngestConfig; CLI flags override its fields.",
)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="File to ingest (Parquet/CSV/JSONL).",
)
@click.option("--dataset", default=None, help="Bronze dataset name.")
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Catalog root [default: data/crucible].",
)
@click.option("--fmt", type=click.Choice(["auto", "parquet", "csv", "jsonl"]), default=None)
@click.option("--batch-size", type=int, default=None)
@click.option(
    "--via-stream",
    is_flag=True,
    default=False,
    help="Replay a JSONL input through the in-memory broker (exercises the streaming path).",
)
def ingest(
    config_path: Path | None,
    input_path: Path | None,
    dataset: str | None,
    root: Path | None,
    fmt: str | None,
    batch_size: int | None,
    via_stream: bool,
) -> None:
    """Land a source into the bronze layer, idempotently."""
    cfg = load_config(
        IngestConfig,
        config_path,
        input=input_path,
        dataset=dataset,
        root=root,
        fmt=fmt,
        batch_size=batch_size,
        via_stream=via_stream or None,
    )
    catalog = Catalog(cfg.root)
    source_name = cfg.source_name or cfg.input.name
    if cfg.via_stream:
        if cfg.input.suffix not in (".jsonl", ".json"):
            raise click.UsageError("--via-stream replays JSONL inputs only")
        broker = InMemoryBroker()
        replay_jsonl(cfg.input, broker, cfg.dataset)
        source: object = StreamSource(broker, cfg.dataset, batch_size=cfg.batch_size)
    else:
        source = open_source(cfg.input, cfg.fmt, cfg.batch_size)
    result = land(source, catalog, cfg.dataset, source_name)  # type: ignore[arg-type]
    click.echo(json.dumps(result.as_dict(), indent=2))


@main.command()
@click.argument("query")
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/crucible"),
    show_default=True,
)
def sql(query: str, root: Path) -> None:
    """Run SQL over catalog views (e.g. bronze_synth); prints JSON rows."""
    click.echo(json.dumps(Catalog(root).query(query), indent=2, default=str))


@main.command()
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/crucible"),
    show_default=True,
)
def catalog(root: Path) -> None:
    """Show per-layer datasets with part and row counts."""
    click.echo(json.dumps(Catalog(root).summary(), indent=2))


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML QualityConfig (see configs/quality_default.yaml).",
)
@click.option("--dataset", required=True, help="Bronze dataset to gate.")
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/crucible"),
    show_default=True,
)
def promote(config_path: Path | None, dataset: str, root: Path) -> None:
    """Run the quality gate: bronze -> silver, failures -> quarantine.

    Writes JSON+Markdown reports under <root>/reports/quality/ and exits
    nonzero when promotion is blocked.
    """
    cfg = load_config(QualityConfig, config_path)
    result = run_gate(Catalog(root), dataset, cfg)
    write_report(result, cfg, root)
    click.echo(json.dumps(result.as_dict(), indent=2))
    if result.verdict == "blocked":
        raise SystemExit(1)


@main.command()
@click.option("--dataset", required=True)
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/crucible"),
    show_default=True,
)
def versions(dataset: str, root: Path) -> None:
    """List version snapshots for a dataset (stage, id, output hash, rows)."""
    records = list_snapshots(root, dataset)
    click.echo(
        json.dumps(
            [
                {
                    "snapshot_id": record["snapshot_id"],
                    "stage": record["stage"],
                    "created_at": record["created_at"],
                    "code_version": record["code_version"],
                    "output_hash": str(record["output"]["content_hash"])[:12],
                    "n_rows": record["output"]["n_rows"],
                }
                for record in records
            ],
            indent=2,
        )
    )


@main.command(name="verify-snapshot")
@click.option("--dataset", required=True)
@click.option("--snapshot-id", default=None, help="Defaults to the newest snapshot.")
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/crucible"),
    show_default=True,
)
def verify_snapshot_cmd(dataset: str, snapshot_id: str | None, root: Path) -> None:
    """Check that on-disk content still matches a pinned snapshot; exit 1 if not."""
    records = list_snapshots(root, dataset)
    if snapshot_id is not None:
        records = [r for r in records if r["snapshot_id"] == snapshot_id]
    if not records:
        raise click.UsageError(f"no snapshots found for {dataset!r}")
    record = records[-1]
    ok, detail = verify_snapshot(Catalog(root), record)
    click.echo(
        json.dumps(
            {
                "snapshot_id": record["snapshot_id"],
                "stage": record["stage"],
                "ok": ok,
                "detail": detail,
            },
            indent=2,
        )
    )
    if not ok:
        raise SystemExit(1)


@main.command()
@click.option("--dataset", default=None, help="Show upstream ancestry of e.g. silver/synth.")
@click.option("--mermaid", is_flag=True, default=False, help="Print a Mermaid diagram.")
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/crucible"),
    show_default=True,
)
def lineage(dataset: str | None, mermaid: bool, root: Path) -> None:
    """Show the lineage graph recorded by ingest/promote/dedup runs."""
    graph = LineageGraph.from_root(root)
    if mermaid:
        click.echo(graph.to_mermaid())
        return
    payload: dict[str, object] = {
        "datasets": sorted(graph.datasets),
        "jobs": sorted(graph.jobs),
        "edges": [f"{a} -> {b}" for a, b in graph.edges()],
    }
    if dataset is not None:
        payload["upstream_of_" + dataset] = sorted(graph.upstream(dataset))
    click.echo(json.dumps(payload, indent=2))


@main.command(name="dedup")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML DedupConfig (see configs/dedup_default.yaml).",
)
@click.option("--dataset", required=True, help="Silver dataset to deduplicate in place.")
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/crucible"),
    show_default=True,
)
def dedup_cmd(config_path: Path | None, dataset: str, root: Path) -> None:
    """Deduplicate silver: exact + MinHash/LSH near-dups, earliest kept.

    Writes JSON+Markdown reports (the JSON records removed ids for
    evaluation-only scoring).
    """
    cfg = load_config(DedupConfig, config_path)
    result = run_dedup(Catalog(root), dataset, cfg)
    write_dedup_report(result, root)
    summary = result.as_dict()
    summary["removed_ids"] = f"({len(result.removed_ids)} ids; see reports/dedup/{dataset}.json)"
    click.echo(json.dumps(summary, indent=2))


@main.command(name="score-dedup")
@click.option("--dataset", required=True)
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/crucible"),
    show_default=True,
)
@click.option(
    "--sweep",
    default=None,
    help="Comma-separated thresholds (e.g. 0.4,0.5,0.6) to re-cluster and score. "
    "Runs on the reconstructed pre-dedup silver.",
)
def score_dedup_cmd(dataset: str, root: Path, sweep: str | None) -> None:
    """EVALUATION ONLY: score dedup removals against planted gt_dup_of labels.

    The scoring universe is the pre-dedup silver (bronze minus quarantine),
    reconstructed so this works after `crucible dedup` has already rewritten
    silver.
    """
    cat = Catalog(root)
    bronze = cat.read(Layer.BRONZE, dataset)
    quarantined: set[str] = set()
    if cat.parts(Layer.QUARANTINE, dataset):
        quarantined = set(cat.read(Layer.QUARANTINE, dataset).column("id").to_pylist())
    keep = [
        i
        for i, record_id in enumerate(bronze.column("id").to_pylist())
        if record_id not in quarantined
    ]
    pre_dedup = bronze.take(keep)
    try:
        if sweep is not None:
            thresholds = [float(part) for part in sweep.split(",") if part.strip()]
            rows = sweep_dedup_thresholds(pre_dedup, DedupConfig(), thresholds)
            click.echo(json.dumps(rows, indent=2))
            return
        report_path = root / "reports" / "dedup" / f"{dataset}.json"
        if not report_path.exists():
            raise click.UsageError(f"no dedup report at {report_path}; run `crucible dedup` first")
        removed_ids = set(json.loads(report_path.read_text())["removed_ids"])
        click.echo(json.dumps(score_dedup(pre_dedup, removed_ids).as_dict(), indent=2))
    except GroundTruthUnavailable as exc:
        raise click.UsageError(str(exc)) from exc


@main.command(name="score-gate")
@click.option("--dataset", required=True)
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/crucible"),
    show_default=True,
)
def score_gate_cmd(dataset: str, root: Path) -> None:
    """EVALUATION ONLY: score quarantine decisions against planted gt_* labels.

    This is the one CLI path that reads ground truth; it exists so reported
    gate precision/recall are measurements, and it fails cleanly on real
    (non-synthetic) datasets that carry no labels.
    """
    cat = Catalog(root)
    quarantined: set[str] = set()
    if cat.parts(Layer.QUARANTINE, dataset):
        quarantined = set(cat.read(Layer.QUARANTINE, dataset).column("id").to_pylist())
    try:
        score = score_gate(cat.read(Layer.BRONZE, dataset), quarantined)
    except GroundTruthUnavailable as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo(json.dumps(score.as_dict(), indent=2))


@main.command()
@click.option("--dataset", required=True, help="Reference dataset.")
@click.option("--against", required=True, help="Dataset to compare with the reference.")
@click.option(
    "--layer",
    type=click.Choice([layer.value for layer in Layer]),
    default=Layer.BRONZE.value,
    show_default=True,
)
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/crucible"),
    show_default=True,
)
def drift(dataset: str, against: str, layer: str, root: Path) -> None:
    """PSI drift between two datasets' source mix and length distribution."""
    cat = Catalog(root)
    reference = profile_table(cat.read(Layer(layer), dataset))
    current = profile_table(cat.read(Layer(layer), against))
    click.echo(json.dumps(drift_report(reference, current), indent=2))


@main.command()
@click.option("--dataset", required=True, help="Bronze dataset to process.")
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/crucible"),
    show_default=True,
)
@click.option("--quality-config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dedup-config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--force", is_flag=True, help="Run even when this input/config fingerprint completed."
)
def orchestrate(
    dataset: str,
    root: Path,
    quality_config: Path | None,
    dedup_config: Path | None,
    force: bool,
) -> None:
    """Run the idempotent promote -> dedup DAG with retries and metrics."""
    result = run_pipeline(
        root,
        dataset,
        load_config(QualityConfig, quality_config),
        load_config(DedupConfig, dedup_config),
        force=force,
    )
    click.echo(json.dumps(result.as_dict(), indent=2))


@main.command(name="runs")
@click.option(
    "--root", type=click.Path(file_okay=False, path_type=Path), default=Path("data/crucible")
)
@click.option("--limit", type=click.IntRange(1, 1000), default=100, show_default=True)
def runs_cmd(root: Path, limit: int) -> None:
    """List durable orchestration run records."""
    click.echo(json.dumps(list_runs(root, limit), indent=2))


@main.command(name="metrics")
@click.option(
    "--root", type=click.Path(file_okay=False, path_type=Path), default=Path("data/crucible")
)
@click.option("--run-id", default=None)
@click.option("--limit", type=click.IntRange(1, 1000), default=100, show_default=True)
def metrics_cmd(root: Path, run_id: str | None, limit: int) -> None:
    """List stage duration and throughput metrics."""
    click.echo(json.dumps(MetricsStore(root).list(run_id=run_id, limit=limit), indent=2))


@main.command()
@click.option(
    "--root", type=click.Path(file_okay=False, path_type=Path), default=Path("data/crucible")
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=click.IntRange(1, 65535), default=8000, show_default=True)
def serve(root: Path, host: str, port: int) -> None:
    """Serve the read-only metadata API (requires the serve extra)."""
    try:
        import uvicorn

        from crucible.serve import create_app
    except (ImportError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    uvicorn.run(create_app(root), host=host, port=port)


@main.command()
@click.option("--n-docs", type=int, default=5000, show_default=True)
@click.option("--seq-len", type=int, default=256, show_default=True)
@click.option("--train-steps", type=int, default=30, show_default=True)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("benchmarks/results"),
    show_default=True,
)
def bench(n_docs: int, seq_len: int, train_steps: int, out_dir: Path) -> None:
    """Measure end-to-end stage throughput; writes a JSON report."""
    click.echo(
        json.dumps(
            run_bench(n_docs=n_docs, seq_len=seq_len, train_steps=train_steps, out_dir=out_dir),
            indent=2,
        )
    )


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="YAML ExperimentConfig under configs/experiments/.",
)
@click.option(
    "--out",
    "output_root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results/experiments"),
    show_default=True,
)
def assay(config_path: Path, output_root: Path) -> None:
    """Run a content-addressed, multi-seed data-centric experiment."""
    cfg = load_config(ExperimentConfig, config_path)
    study = STUDIES.get(cfg.study)
    if study is None:
        raise click.UsageError(f"unknown study {cfg.study!r}; choose from {sorted(STUDIES)}")
    result = run_experiment(cfg, study, output_root)
    click.echo(
        json.dumps(
            {
                "study": result.study,
                "config_hash": result.config_hash,
                "result_hash": result.result_hash,
                "artifact_dir": str(result.artifact_dir),
                "rows": len(result.rows),
            },
            indent=2,
        )
    )


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="YAML ForecastRunConfig (see configs/forecast_financial.yaml).",
)
@click.option(
    "--out",
    "output_root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results/forecast"),
    show_default=True,
)
def forecast(config_path: Path, output_root: Path) -> None:
    """Train and evaluate a probabilistic time-series forecaster."""
    from crucible.forecast import ForecastRunConfig, run_forecast

    try:
        result = run_forecast(load_config(ForecastRunConfig, config_path), output_root)
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result.as_dict(), indent=2))


@main.command()
@click.option(
    "--workdir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to write smoke artifacts (default: a temp dir).",
)
def smoke(workdir: Path | None) -> None:
    """Run the offline end-to-end smoke check; exits nonzero on failure."""
    try:
        report = run_smoke(workdir)
    except SmokeFailure as failure:
        click.echo(json.dumps({"ok": False, "error": str(failure)}, indent=2))
        sys.exit(1)
    click.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
