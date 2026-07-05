import json
from pathlib import Path

from click.testing import CliRunner

from crucible import __version__
from crucible.cli import main


def test_version() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_synth_writes_outputs(tmp_path: Path) -> None:
    out = tmp_path / "raw"
    result = CliRunner().invoke(main, ["synth", "--out", str(out), "--n-docs", "50", "--seed", "9"])
    assert result.exit_code == 0, result.output
    assert (out / "corpus.jsonl").exists()
    assert (out / "corpus.parquet").exists()
    report = json.loads((out / "generation_report.json").read_text())
    assert report["n_records"] == 50
    assert report["seed"] == 9


def test_smoke_passes(tmp_path: Path) -> None:
    result = CliRunner().invoke(main, ["smoke", "--workdir", str(tmp_path / "smoke")])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["ok"] is True
    assert "determinism" in report["checks_passed"]
    assert "bronze_stream_fallback_equivalent" in report["checks_passed"]


def _synth_corpus(tmp_path: Path) -> Path:
    runner = CliRunner()
    out = tmp_path / "raw"
    result = runner.invoke(
        main, ["synth", "--out", str(out), "--n-docs", "60", "--seed", "3", "--fmt", "jsonl"]
    )
    assert result.exit_code == 0, result.output
    return out / "corpus.jsonl"


def test_ingest_sql_catalog_round_trip(tmp_path: Path) -> None:
    corpus = _synth_corpus(tmp_path)
    root = tmp_path / "catalog"
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["ingest", "--input", str(corpus), "--dataset", "synth", "--root", str(root)],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["rows_written"] == 60

    # Idempotent from the CLI too.
    result = runner.invoke(
        main,
        ["ingest", "--input", str(corpus), "--dataset", "synth", "--root", str(root)],
    )
    assert json.loads(result.output)["parts_written"] == 0

    result = runner.invoke(
        main, ["sql", "SELECT count(*) AS n FROM bronze_synth", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [{"n": 60}]

    result = runner.invoke(main, ["catalog", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["bronze"]["synth"]["rows"] == 60


def test_ingest_via_stream(tmp_path: Path) -> None:
    corpus = _synth_corpus(tmp_path)
    root = tmp_path / "catalog"
    result = CliRunner().invoke(
        main,
        [
            "ingest",
            "--input",
            str(corpus),
            "--dataset",
            "synth_s",
            "--root",
            str(root),
            "--via-stream",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["rows_written"] == 60


def test_ingest_via_stream_rejects_non_jsonl(tmp_path: Path) -> None:
    csv = tmp_path / "rows.csv"
    csv.write_text("id\na\n")
    result = CliRunner().invoke(
        main,
        ["ingest", "--input", str(csv), "--dataset", "d", "--root", str(tmp_path), "--via-stream"],
    )
    assert result.exit_code != 0
    assert "JSONL inputs only" in result.output
