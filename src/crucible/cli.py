"""The ``crucible`` command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from crucible import __version__
from crucible.config import load_config
from crucible.ingest import (
    IngestConfig,
    InMemoryBroker,
    StreamSource,
    land,
    open_source,
    replay_jsonl,
)
from crucible.smoke import SmokeFailure, run_smoke
from crucible.storage import Catalog
from crucible.synth import (
    SynthConfig,
    generate_corpus,
    generation_report,
    write_jsonl,
    write_parquet,
)


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
