from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from crucible.forecast.metrics import (
    MetricError,
    apply_quantile_calibration,
    compare_with_baseline,
    evaluate_forecasts,
    fit_quantile_calibration,
    moving_block_bootstrap_skill,
    persistence_forecast,
    pinball_loss,
    seasonal_naive_forecast,
)
from crucible.forecast.plots import write_evaluation_plots


def test_pinball_and_point_metrics_have_known_values() -> None:
    actual = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    median = np.array([[0.5, -0.5], [1.5, -1.5]], dtype=np.float32)
    predictions = np.stack([median - 1, median, median + 1], axis=2)
    metrics = evaluate_forecasts(
        actual,
        predictions,
        (0.1, 0.5, 0.9),
        training_target=np.array([0.0, 1.0, 0.0]),
    )
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(0.5)
    assert metrics["directional_accuracy"] == 1.0
    assert metrics["interval_coverage"] == 1.0
    assert metrics["quantile_crossing_rate"] == 0.0
    assert len(metrics["per_horizon"]) == 2
    assert pinball_loss(actual, actual, 0.5) == 0


def test_baseline_and_metric_contracts() -> None:
    inputs = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    baseline = persistence_forecast(inputs, 2, 1)
    np.testing.assert_array_equal(baseline, [[10, 10], [22, 22]])
    np.testing.assert_array_equal(
        seasonal_naive_forecast(inputs, 2, 1, seasonality=2), [[7, 10], [19, 22]]
    )
    assert compare_with_baseline({"mae": 1.0}, {"mae": 2.0})["mae_skill_score"] == 0.5
    with pytest.raises(MetricError):
        pinball_loss(np.ones(2), np.ones(3), 0.5)
    with pytest.raises(MetricError, match="median"):
        evaluate_forecasts(
            np.ones((2, 1)),
            np.ones((2, 1, 2)),
            (0.1, 0.9),
            training_target=np.ones(3),
        )


def test_validation_only_quantile_calibration_is_ordered() -> None:
    actual = np.linspace(-1, 1, 200).reshape(100, 2)
    predictions = np.zeros((100, 2, 3))
    corrections = fit_quantile_calibration(actual, predictions, (0.1, 0.5, 0.9))
    calibrated = apply_quantile_calibration(predictions, corrections)
    assert calibrated.shape == predictions.shape
    assert np.all(np.diff(calibrated, axis=2) >= 0)
    assert np.mean(actual <= calibrated[:, :, 1]) == pytest.approx(0.5, abs=0.02)
    with pytest.raises(MetricError, match="shapes"):
        fit_quantile_calibration(actual, predictions[:, :, :2], (0.1, 0.5, 0.9))
    with pytest.raises(MetricError, match="corrections"):
        apply_quantile_calibration(predictions, np.ones(2))


def test_metric_edge_cases_are_explicit() -> None:
    actual = np.ones((2, 2))
    predictions = np.ones((2, 2, 3))
    metrics = evaluate_forecasts(actual, predictions, (0.1, 0.5, 0.9), training_target=np.ones(3))
    assert metrics["mase"] is None
    with pytest.raises(MetricError, match="actual"):
        evaluate_forecasts(actual[:, :1], predictions, (0.1, 0.5, 0.9), training_target=np.ones(3))
    with pytest.raises(MetricError, match="sorted"):
        evaluate_forecasts(actual, predictions, (0.5, 0.1, 0.9), training_target=np.ones(3))
    with pytest.raises(MetricError, match="enough"):
        seasonal_naive_forecast(np.ones((2, 2, 1)), 1, 0, seasonality=3)


def test_moving_block_bootstrap_is_paired_deterministic_and_positive() -> None:
    actual = np.sin(np.arange(120)[:, None] / 7)
    model = actual + 0.1
    baseline = actual + 0.3
    first = moving_block_bootstrap_skill(
        actual, model, baseline, block_length=5, samples=200, seed=2
    )
    second = moving_block_bootstrap_skill(
        actual, model, baseline, block_length=5, samples=200, seed=2
    )
    assert first == second
    assert first["ci_low"] > 0
    with pytest.raises(MetricError, match="shapes"):
        moving_block_bootstrap_skill(actual, model[:-1], baseline, block_length=5, samples=200)


def test_plots_are_parseable_held_out_evaluation_artifacts(tmp_path: Path) -> None:
    actual = np.sin(np.arange(40)[:, None] / 4)
    median = actual + 0.05
    predictions = np.stack([median - 0.2, median, median + 0.2], axis=2)
    paths = write_evaluation_plots(
        tmp_path,
        actual=actual,
        predictions=predictions,
        baseline=actual + 0.1,
        quantiles=(0.1, 0.5, 0.9),
        history=[
            {"epoch": 1.0, "train_loss": 1.0, "validation_loss": 1.2},
            {"epoch": 2.0, "train_loss": 0.7, "validation_loss": 0.8},
        ],
    )
    assert set(paths) == {
        "forecast",
        "residuals",
        "training",
        "calibration",
        "metrics_by_horizon",
    }
    for value in paths.values():
        root = ElementTree.parse(value).getroot()
        assert root.tag.endswith("svg")
        text = Path(value).read_text().lower()
        assert "nan" not in text and "inf" not in text
