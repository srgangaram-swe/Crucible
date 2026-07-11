import json
from pathlib import Path

import pytest

from crucible.assay.harness import ExperimentConfig, bootstrap_ci, run_experiment


def test_config_hash_is_canonical_and_sensitive() -> None:
    first = ExperimentConfig(study="demo", seeds=[1], parameters={"a": 1, "b": 2})
    reordered = ExperimentConfig(study="demo", seeds=[1], parameters={"b": 2, "a": 1})
    changed = first.model_copy(update={"seeds": [2]})
    assert first.config_hash == reordered.config_hash
    assert first.config_hash != changed.config_hash


def test_bootstrap_ci_is_deterministic_and_validates() -> None:
    assert bootstrap_ci([2.5], samples=100, seed=1) == (2.5, 2.5)
    first = bootstrap_ci([1.0, 2.0, 3.0], samples=500, seed=7)
    assert first == bootstrap_ci([1.0, 2.0, 3.0], samples=500, seed=7)
    assert first[0] <= 2.0 <= first[1]
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_ci([], samples=100, seed=1)


def test_run_experiment_writes_content_addressed_artifacts(tmp_path: Path) -> None:
    config = ExperimentConfig(study="demo", seeds=[1, 2], bootstrap_samples=100)

    def study(cfg: ExperimentConfig, seed: int) -> list[dict[str, object]]:
        return [{"arm": "control", "seed": seed, "validation_loss": 3.0 + seed}]

    result = run_experiment(config, study, tmp_path)
    assert result.artifact_dir.parent.name == config.config_hash
    assert result.artifact_dir.name == result.result_hash
    assert {path.name for path in result.artifact_dir.iterdir()} == {
        "plot.svg",
        "report.md",
        "results.csv",
        "results.json",
    }
    payload = json.loads((result.artifact_dir / "results.json").read_text())
    assert payload["config_hash"] == config.config_hash
    assert payload["result_hash"] == result.result_hash
    assert payload["summary"][0]["n_seeds"] == 2
    assert config.config_hash in (result.artifact_dir / "plot.svg").read_text()
    assert "<line" in (result.artifact_dir / "plot.svg").read_text()


def test_run_experiment_rejects_invalid_study(tmp_path: Path) -> None:
    config = ExperimentConfig(study="empty", seeds=[1], bootstrap_samples=100)
    with pytest.raises(ValueError, match="arm and seed"):
        run_experiment(config, lambda cfg, seed: [], tmp_path)
