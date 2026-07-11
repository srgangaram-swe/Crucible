"""Point, probabilistic, directional, and baseline forecast evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class MetricError(ValueError):
    """Shapes or quantiles do not satisfy the forecast metric contract."""


def fit_quantile_calibration(
    actual: np.ndarray,
    predictions: np.ndarray,
    quantiles: tuple[float, ...],
) -> np.ndarray:
    """Fit additive quantile corrections on validation predictions only."""
    if predictions.shape[:2] != actual.shape or predictions.shape[2] != len(quantiles):
        raise MetricError("calibration shapes do not match quantiles")
    return np.array(
        [
            np.quantile(actual - predictions[:, :, index], quantile)
            for index, quantile in enumerate(quantiles)
        ],
        dtype=np.float32,
    )


def apply_quantile_calibration(predictions: np.ndarray, corrections: np.ndarray) -> np.ndarray:
    if predictions.ndim != 3 or predictions.shape[2] != len(corrections):
        raise MetricError("calibration corrections do not match prediction quantiles")
    corrected = predictions + corrections.reshape(1, 1, -1)
    return np.maximum.accumulate(corrected, axis=2).astype(np.float32)


def pinball_loss(actual: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    if actual.shape != prediction.shape or not 0 < quantile < 1:
        raise MetricError("pinball inputs must share a shape and quantile must be in (0, 1)")
    error = actual - prediction
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def persistence_forecast(
    inputs_original: np.ndarray, horizon: int, target_index: int
) -> np.ndarray:
    if inputs_original.ndim != 3 or horizon < 1:
        raise MetricError("persistence inputs must be [sample, context, feature]")
    return np.repeat(inputs_original[:, -1, target_index, None], horizon, axis=1)


def seasonal_naive_forecast(
    inputs_original: np.ndarray,
    horizon: int,
    target_index: int,
    seasonality: int,
) -> np.ndarray:
    if inputs_original.ndim != 3 or seasonality < 1 or inputs_original.shape[1] < seasonality:
        raise MetricError("seasonal baseline requires enough [sample, context, feature] history")
    indices = [
        inputs_original.shape[1] - seasonality + (step % seasonality) for step in range(horizon)
    ]
    return inputs_original[:, indices, target_index]


def evaluate_forecasts(
    actual: np.ndarray,
    quantile_predictions: np.ndarray,
    quantiles: tuple[float, ...],
    *,
    training_target: np.ndarray,
) -> dict[str, Any]:
    """Evaluate direct multi-horizon predictions in original target units."""
    if actual.ndim != 2 or quantile_predictions.shape[:2] != actual.shape:
        raise MetricError("actual must be [sample, horizon] and predictions [sample, horizon, q]")
    if quantile_predictions.shape[2] != len(quantiles) or tuple(sorted(quantiles)) != quantiles:
        raise MetricError("prediction quantiles must be sorted and match the final dimension")
    if 0.5 not in quantiles or len(training_target) < 2:
        raise MetricError("metrics require a median quantile and at least two training targets")
    median = quantile_predictions[:, :, quantiles.index(0.5)]
    error = median - actual
    absolute = np.abs(error)
    mae = float(np.mean(absolute))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denominator = np.abs(actual) + np.abs(median)
    smape = float(
        np.mean(
            np.divide(
                2 * absolute, denominator, out=np.zeros_like(error), where=denominator > 1e-12
            )
        )
    )
    naive_scale = float(np.mean(np.abs(np.diff(training_target))))
    mase = mae / naive_scale if naive_scale > 1e-12 else None
    directional = float(np.mean(np.sign(median) == np.sign(actual)))
    losses = {
        f"q{quantile:g}": pinball_loss(actual, quantile_predictions[:, :, i], quantile)
        for i, quantile in enumerate(quantiles)
    }
    low = quantile_predictions[:, :, 0]
    high = quantile_predictions[:, :, -1]
    coverage = float(np.mean((actual >= low) & (actual <= high)))
    interval_width = float(np.mean(high - low))
    crossing = float(np.mean(np.diff(quantile_predictions, axis=2) < 0))
    per_horizon = [
        {
            "horizon": index + 1,
            "mae": float(np.mean(np.abs(error[:, index]))),
            "rmse": float(np.sqrt(np.mean(np.square(error[:, index])))),
            "directional_accuracy": float(
                np.mean(np.sign(median[:, index]) == np.sign(actual[:, index]))
            ),
        }
        for index in range(actual.shape[1])
    ]
    return {
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "mase": mase,
        "directional_accuracy": directional,
        "mean_pinball_loss": float(np.mean(list(losses.values()))),
        "pinball_by_quantile": losses,
        "interval_coverage": coverage,
        "interval_width": interval_width,
        "quantile_crossing_rate": crossing,
        "per_horizon": per_horizon,
    }


def compare_with_baseline(
    model_metrics: dict[str, Any], baseline_metrics: dict[str, Any]
) -> dict[str, Any]:
    model_mae = float(model_metrics["mae"])
    baseline_mae = float(baseline_metrics["mae"])
    return {
        "mae_skill_score": 1 - model_mae / baseline_mae if baseline_mae > 0 else None,
        "model_beats_baseline_mae": model_mae < baseline_mae,
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
    }


def moving_block_bootstrap_skill(
    actual: np.ndarray,
    median_prediction: np.ndarray,
    baseline: np.ndarray,
    *,
    block_length: int,
    samples: int = 1_000,
    seed: int = 41,
) -> dict[str, float | int]:
    """Time-block bootstrap CI for paired MAE skill on overlapping horizons."""
    if actual.shape != median_prediction.shape or actual.shape != baseline.shape:
        raise MetricError("bootstrap actual, model, and baseline shapes must match")
    if block_length < 1 or samples < 100 or len(actual) < block_length:
        raise MetricError("bootstrap requires valid block length, samples, and origins")
    model_error = np.mean(np.abs(actual - median_prediction), axis=1)
    baseline_error = np.mean(np.abs(actual - baseline), axis=1)
    rng = np.random.default_rng(seed)
    max_start = len(actual) - block_length + 1
    blocks_needed = math.ceil(len(actual) / block_length)
    skills: list[float] = []
    for _ in range(samples):
        starts = rng.integers(0, max_start, size=blocks_needed)
        indices = np.concatenate([np.arange(start, start + block_length) for start in starts])[
            : len(actual)
        ]
        denominator = float(np.mean(baseline_error[indices]))
        if denominator <= 1e-12:
            raise MetricError("bootstrap baseline error must be positive")
        skills.append(1 - float(np.mean(model_error[indices])) / denominator)
    low, high = np.quantile(skills, [0.025, 0.975])
    observed_denominator = float(np.mean(baseline_error))
    if observed_denominator <= 1e-12:
        raise MetricError("bootstrap baseline error must be positive")
    observed = 1 - float(np.mean(model_error)) / observed_denominator
    return {
        "observed_skill": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "block_length": block_length,
        "samples": samples,
    }
