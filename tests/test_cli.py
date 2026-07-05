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
