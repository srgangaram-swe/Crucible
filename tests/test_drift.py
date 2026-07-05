from pathlib import Path

import pytest

from crucible.quality import DatasetProfile, drift_report, population_stability_index
from crucible.quality.drift import profile_table
from crucible.synth import SynthConfig, generate_corpus, write_parquet

BASE = SynthConfig(seed=7, n_docs=300)
SAME_DIST_OTHER_SEED = SynthConfig(seed=8, n_docs=300)
SKEWED = SynthConfig(
    seed=7,
    n_docs=300,
    domain_weights={"news": 0.05, "forum_qa": 0.05, "code": 0.85, "recipes": 0.05},
)


def _profile(cfg: SynthConfig, tmp_path: Path, name: str) -> DatasetProfile:
    import pyarrow.parquet as pq

    path = tmp_path / f"{name}.parquet"
    write_parquet(generate_corpus(cfg), path)
    return profile_table(pq.read_table(path))


def test_no_drift_between_same_distribution_samples(tmp_path: Path) -> None:
    ref = _profile(BASE, tmp_path, "ref")
    cur = _profile(SAME_DIST_OTHER_SEED, tmp_path, "cur")
    report = drift_report(ref, cur)
    assert report["verdict"] == "none", report


def test_major_drift_detected_on_skewed_mixture(tmp_path: Path) -> None:
    ref = _profile(BASE, tmp_path, "ref")
    cur = _profile(SKEWED, tmp_path, "cur")
    report = drift_report(ref, cur)
    assert report["source_verdict"] == "major", report
    assert report["source_psi"] > 0.25


def test_psi_zero_for_identical_distributions() -> None:
    dist = [0.25, 0.25, 0.5]
    assert population_stability_index(dist, dist) == pytest.approx(0.0, abs=1e-9)


def test_psi_requires_shared_binning() -> None:
    with pytest.raises(ValueError, match="share binning"):
        population_stability_index([0.5, 0.5], [1.0])


def test_profile_round_trips_through_json(tmp_path: Path) -> None:
    ref = _profile(BASE, tmp_path, "ref")
    path = tmp_path / "profile.json"
    ref.to_json(path)
    assert DatasetProfile.from_json(path) == ref


def test_profile_rejects_empty_table() -> None:
    import pyarrow as pa

    with pytest.raises(ValueError, match="empty"):
        profile_table(
            pa.table(
                {"text": pa.array([], type=pa.string()), "source": pa.array([], type=pa.string())}
            )
        )
