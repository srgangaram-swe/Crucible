"""Torch-backed model and end-to-end forecasting tests; run in the CI train job."""

import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crucible.forecast.data import (  # noqa: E402
    SyntheticFinancialConfig,
    prepare_forecast_data,
    synthetic_financial_series,
)
from crucible.forecast.model import PatchTSTConfig, build_patchtst, quantile_loss  # noqa: E402
from crucible.forecast.pipeline import ForecastRunConfig, run_forecast  # noqa: E402
from crucible.forecast.trainer import (  # noqa: E402
    ForecastTrainConfig,
    load_forecaster,
    predict,
    train_forecaster,
)

MODEL = PatchTSTConfig(
    context_length=32,
    horizon=3,
    patch_length=8,
    patch_stride=4,
    d_model=32,
    n_heads=4,
    n_layers=2,
    dropout=0.0,
    quantiles=(0.1, 0.5, 0.9),
    seed=9,
)


def test_model_shapes_ordered_quantiles_gradients_and_validation() -> None:
    model = build_patchtst(MODEL, 4)
    inputs = torch.randn(5, 32, 4)
    targets = torch.randn(5, 3)
    predictions = model(inputs)
    assert predictions.shape == (5, 3, 3)
    assert torch.all(predictions[:, :, 0] <= predictions[:, :, 1])
    assert torch.all(predictions[:, :, 1] <= predictions[:, :, 2])
    loss = quantile_loss(predictions, targets, MODEL.quantiles)
    loss.backward()
    assert loss.item() >= 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    model.eval()
    torch.testing.assert_close(model(inputs), model(inputs), rtol=0, atol=0)
    exact_loss = quantile_loss(torch.zeros(1, 2, 3), torch.ones(1, 2), (0.1, 0.5, 0.9))
    assert exact_loss.item() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(2, 31, 4))
    with pytest.raises(ValueError, match="finite"):
        model(torch.full((2, 32, 4), float("nan")))
    with pytest.raises(ValueError, match="divisible"):
        PatchTSTConfig(d_model=30, n_heads=4)
    with pytest.raises(ValueError, match="quantiles"):
        PatchTSTConfig(quantiles=(0.1, 0.9))


def test_training_is_deterministic_and_checkpoint_reload_matches(tmp_path: Path) -> None:
    frame = synthetic_financial_series(SyntheticFinancialConfig(n_steps=360, seed=9))
    bundle = prepare_forecast_data(frame, context_length=32, horizon=3, embargo=3)
    training = ForecastTrainConfig(
        epochs=4,
        batch_size=32,
        learning_rate=0.002,
        warmup_epochs=1,
        patience=4,
        seed=9,
        device="cpu",
    )
    first, first_result = train_forecaster(
        MODEL, training, bundle.train, bundle.validation, tmp_path / "first.pt"
    )
    second, second_result = train_forecaster(
        MODEL, training, bundle.train, bundle.validation, tmp_path / "second.pt"
    )
    assert first_result.history == second_result.history
    first_predictions = predict(first, bundle.test, 64, "cpu")
    second_predictions = predict(second, bundle.test, 64, "cpu")
    np.testing.assert_array_equal(first_predictions, second_predictions)
    loaded, metadata = load_forecaster(tmp_path / "first.pt")
    np.testing.assert_array_equal(first_predictions, predict(loaded, bundle.test, 64, "cpu"))
    assert metadata["epoch"] == first_result.best_epoch
    assert metadata["dataset_hash"]
    assert first_result.history[-1]["gradient_norm"] >= 0


def test_forecast_pipeline_trains_evaluates_and_writes_artifacts(tmp_path: Path) -> None:
    config = ForecastRunConfig(
        synthetic=SyntheticFinancialConfig(n_steps=420, seed=12),
        model=MODEL.model_copy(update={"seed": 12}),
        training=ForecastTrainConfig(
            epochs=8,
            batch_size=32,
            learning_rate=0.002,
            warmup_epochs=1,
            patience=5,
            seed=12,
            device="cpu",
        ),
        embargo=3,
    )
    result = run_forecast(config, tmp_path)
    assert result.metrics["model"]["quantile_crossing_rate"] == 0.0
    assert np.isfinite(result.metrics["model"]["mean_pinball_loss"])
    expected = {
        "best_model.pt",
        "calibration.svg",
        "forecast.svg",
        "history.json",
        "metrics.json",
        "metrics_by_horizon.svg",
        "model_card.json",
        "predictions.csv",
        "residuals.svg",
        "run.json",
        "training.svg",
    }
    assert expected <= {path.name for path in result.artifact_dir.iterdir()}
    run = json.loads((result.artifact_dir / "run.json").read_text())
    assert run["split"]["embargo"] == 3
    assert run["scaler"]["fitted_through"] == run["split"]["train_end"]
    for svg in result.artifact_dir.glob("*.svg"):
        assert ElementTree.parse(svg).getroot().tag.endswith("svg")
