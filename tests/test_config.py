from pathlib import Path

import pytest

from crucible.config import load_config, load_yaml
from crucible.synth import SynthConfig


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        load_yaml(path)


def test_load_yaml_empty_file_is_empty_mapping(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_yaml(path) == {}


def test_load_config_overrides_ignore_none(tmp_path: Path) -> None:
    path = tmp_path / "synth.yaml"
    path.write_text("seed: 7\nn_docs: 50\n")
    cfg = load_config(SynthConfig, path, seed=None, n_docs=99)
    assert cfg.seed == 7  # None override did not clobber the file value
    assert cfg.n_docs == 99  # explicit override wins


def test_load_config_without_file_uses_defaults() -> None:
    cfg = load_config(SynthConfig, None)
    assert cfg.n_docs == SynthConfig().n_docs
