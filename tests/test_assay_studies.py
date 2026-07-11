from crucible.assay.harness import ExperimentConfig
from crucible.assay.studies import STUDIES, _texts


def _config(study: str) -> ExperimentConfig:
    parameters = {"token_scales": [1000, 2000, 3000]} if study == "scaling_law" else {}
    return ExperimentConfig(
        study=study,
        seeds=[3],
        n_docs=80,
        train_tokens=2000,
        bootstrap_samples=100,
        parameters=parameters,
    )


def test_every_study_is_deterministic_and_equal_compute() -> None:
    for name, study in STUDIES.items():
        config = _config(name)
        first = study(config, 3)
        assert first == study(config, 3)
        assert first
        assert {row["train_tokens"] for row in first} <= {1000, 2000, 3000}
        assert all(float(row["validation_loss"]) > 0 for row in first)


def test_study_contracts() -> None:
    quality = STUDIES["quality_ablation"](_config("quality_ablation"), 3)
    assert {row["arm"] for row in quality} == {"ungated", "default", "repeated_sentences"}
    assert all("keep_rate" in row for row in quality)

    mixture = STUDIES["mixture_ablation"](_config("mixture_ablation"), 3)
    assert {row["selection"] for row in mixture} == {"grid", "proxy"}
    assert any(str(row["arm"]).startswith("proxy_selected_") for row in mixture)

    scaling = STUDIES["scaling_law"](_config("scaling_law"), 3)
    assert len({row["fitted_log_slope"] for row in scaling}) == 1


def test_training_and_validation_ids_are_disjoint() -> None:
    training, validation = _texts(_config("quality_ablation"), 3)
    assert {row["id"] for row in training}.isdisjoint(row["id"] for row in validation)
