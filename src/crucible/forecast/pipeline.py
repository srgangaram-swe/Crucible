"""End-to-end forecast training, sealed evaluation, and artifact publication."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from crucible import __version__
from crucible.forecast.data import (
    ForecastDataBundle,
    SyntheticFinancialConfig,
    TimeSeriesFrame,
    prepare_forecast_data,
    read_csv_series,
    synthetic_financial_series,
)
from crucible.forecast.metrics import (
    apply_quantile_calibration,
    compare_with_baseline,
    evaluate_forecasts,
    fit_quantile_calibration,
    moving_block_bootstrap_skill,
    persistence_forecast,
    seasonal_naive_forecast,
)
from crucible.forecast.model import PatchTSTConfig
from crucible.forecast.plots import write_evaluation_plots
from crucible.forecast.trainer import (
    ForecastTrainConfig,
    predict,
    train_forecaster,
    write_history,
)
from crucible.utils.hashing import canonical_json, sha256_texts


class ForecastRunConfig(BaseModel):
    """One reproducible, sealed train/validation/test forecast run."""

    input_csv: Path | None = None
    timestamp_column: str = "timestamp"
    target_column: str = "return"
    feature_columns: list[str] | None = None
    synthetic: SyntheticFinancialConfig = Field(default_factory=SyntheticFinancialConfig)
    model: PatchTSTConfig = Field(default_factory=PatchTSTConfig)
    training: ForecastTrainConfig = Field(default_factory=ForecastTrainConfig)
    train_fraction: float = Field(default=0.7, gt=0, lt=1)
    validation_fraction: float = Field(default=0.15, gt=0, lt=1)
    embargo: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _split_capacity(self) -> ForecastRunConfig:
        if self.train_fraction + self.validation_fraction >= 0.95:
            raise ValueError("train + validation fractions must leave at least 5% for test")
        return self


@dataclass(frozen=True, slots=True)
class ForecastRunResult:
    run_id: str
    artifact_dir: Path
    metrics: dict[str, Any]
    plot_paths: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "artifact_dir": str(self.artifact_dir),
            "metrics": self.metrics,
            "plots": self.plot_paths,
        }


def _frame(config: ForecastRunConfig) -> tuple[TimeSeriesFrame, str]:
    if config.input_csv is None:
        frame = synthetic_financial_series(config.synthetic)
        return frame, "synthetic_regime_switching_returns"
    frame = read_csv_series(
        config.input_csv,
        timestamp_column=config.timestamp_column,
        target_column=config.target_column,
        feature_columns=config.feature_columns,
    )
    return frame, str(config.input_csv)


def _data_hash(frame: TimeSeriesFrame) -> str:
    return sha256_texts(
        [
            canonical_json(frame.columns),
            frame.timestamps.astype("datetime64[ns]").astype(np.int64).tobytes().hex(),
            frame.values.tobytes().hex(),
        ]
    )


def _inverse_bundle(bundle: ForecastDataBundle) -> tuple[np.ndarray, np.ndarray]:
    actual = bundle.scaler.inverse_target(bundle.test.targets, bundle.target_index)
    inputs = bundle.test.inputs * bundle.scaler.scale + bundle.scaler.mean
    return actual, inputs


def _prediction_rows(
    frame: TimeSeriesFrame,
    bundle: ForecastDataBundle,
    actual: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
    quantiles: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample, origin in enumerate(bundle.test.origins):
        for horizon in range(actual.shape[1]):
            row: dict[str, Any] = {
                "forecast_origin": str(frame.timestamps[origin - 1]),
                "target_timestamp": str(bundle.test.forecast_timestamps[sample, horizon]),
                "horizon": horizon + 1,
                "actual": float(actual[sample, horizon]),
                "persistence": float(baseline[sample, horizon]),
            }
            row.update(
                {
                    f"q{quantile:g}": float(predictions[sample, horizon, index])
                    for index, quantile in enumerate(quantiles)
                }
            )
            rows.append(row)
    return rows


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_forecast(config: ForecastRunConfig, output_root: Path) -> ForecastRunResult:
    """Train once, restore best validation weights, then open the sealed test split."""
    frame, source = _frame(config)
    model_config = config.model.model_copy(
        update={"target_index": frame.target_index, "seed": config.training.seed}
    )
    bundle = prepare_forecast_data(
        frame,
        context_length=model_config.context_length,
        horizon=model_config.horizon,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        embargo=config.embargo,
    )
    data_hash = _data_hash(frame)
    config_payload = config.model_dump(mode="json")
    run_id = sha256_texts([canonical_json(config_payload), data_hash, f"crucible/{__version__}"])[
        :16
    ]
    artifact_dir = Path(output_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model, training = train_forecaster(
        model_config,
        config.training,
        bundle.train,
        bundle.validation,
        artifact_dir / "best_model.pt",
    )
    scaled_validation_predictions = predict(
        model, bundle.validation, config.training.batch_size, training.device
    )
    scaled_predictions = predict(model, bundle.test, config.training.batch_size, training.device)
    validation_predictions = bundle.scaler.inverse_target(
        scaled_validation_predictions, bundle.target_index
    )
    validation_actual = bundle.scaler.inverse_target(bundle.validation.targets, bundle.target_index)
    quantiles = model_config.quantiles
    calibration = fit_quantile_calibration(validation_actual, validation_predictions, quantiles)
    predictions = apply_quantile_calibration(
        bundle.scaler.inverse_target(scaled_predictions, bundle.target_index), calibration
    )
    actual, original_inputs = _inverse_bundle(bundle)
    persistence = persistence_forecast(original_inputs, model_config.horizon, bundle.target_index)
    seasonal = seasonal_naive_forecast(
        original_inputs,
        model_config.horizon,
        bundle.target_index,
        seasonality=min(5, model_config.context_length),
    )
    zero = np.zeros_like(persistence)
    model_metrics = evaluate_forecasts(
        actual,
        predictions,
        quantiles,
        training_target=bundle.training_target_original,
    )
    persistence_metrics = evaluate_forecasts(
        actual,
        np.repeat(persistence[:, :, None], len(quantiles), axis=2),
        quantiles,
        training_target=bundle.training_target_original,
    )
    zero_metrics = evaluate_forecasts(
        actual,
        np.repeat(zero[:, :, None], len(quantiles), axis=2),
        quantiles,
        training_target=bundle.training_target_original,
    )
    seasonal_metrics = evaluate_forecasts(
        actual,
        np.repeat(seasonal[:, :, None], len(quantiles), axis=2),
        quantiles,
        training_target=bundle.training_target_original,
    )
    metrics = {
        "model": model_metrics,
        "persistence": persistence_metrics,
        "zero_return": zero_metrics,
        "seasonal_naive": seasonal_metrics,
        "comparison_to_persistence": compare_with_baseline(model_metrics, persistence_metrics),
        "comparison_to_zero_return": compare_with_baseline(model_metrics, zero_metrics),
        "comparison_to_seasonal_naive": compare_with_baseline(model_metrics, seasonal_metrics),
        "persistence_mae_skill_block_bootstrap": moving_block_bootstrap_skill(
            actual,
            predictions[:, :, quantiles.index(0.5)],
            persistence,
            block_length=max(model_config.horizon, 5),
            samples=1_000,
            seed=config.training.seed,
        ),
    }
    rows = _prediction_rows(frame, bundle, actual, predictions, persistence, quantiles)
    _write_predictions(artifact_dir / "predictions.csv", rows)
    write_history(artifact_dir / "history.json", training)
    _atomic_json(artifact_dir / "metrics.json", metrics)
    _atomic_json(
        artifact_dir / "run.json",
        {
            "run_id": run_id,
            "code_version": __version__,
            "data_hash": data_hash,
            "data_source": source,
            "config": config_payload,
            "resolved_model_config": model_config.model_dump(mode="json"),
            "split": bundle.split.as_dict(),
            "scaler": {
                "mean": bundle.scaler.mean.tolist(),
                "scale": bundle.scaler.scale.tolist(),
                "fitted_through": bundle.scaler.fitted_through,
            },
            "quantile_calibration": calibration.tolist(),
            "columns": bundle.columns,
            "train_windows": len(bundle.train.targets),
            "validation_windows": len(bundle.validation.targets),
            "test_windows": len(bundle.test.targets),
            "training": training.as_dict(),
        },
    )
    _atomic_json(
        artifact_dir / "model_card.json",
        {
            "architecture": "PatchTST-style shared channel encoder with attention pooling",
            "objective": "direct multi-horizon quantile forecasting",
            "intended_use": "offline research and integration validation",
            "not_for": "live trading, investment advice, or unvalidated financial decisions",
            "data": source,
            "limitations": [
                "synthetic benchmark is not evidence of market alpha",
                "no transaction costs, corporate actions, or exchange calendar",
                "past-observed covariates only; known-future features are unsupported",
            ],
        },
    )
    plots = write_evaluation_plots(
        artifact_dir,
        actual=actual,
        predictions=predictions,
        baseline=persistence,
        quantiles=quantiles,
        history=training.history,
    )
    return ForecastRunResult(run_id, artifact_dir, metrics, plots)
