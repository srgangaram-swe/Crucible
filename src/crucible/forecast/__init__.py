"""Financial-ready probabilistic time-series forecasting."""

from crucible.forecast.data import (
    ForecastDataBundle,
    SyntheticFinancialConfig,
    TimeSeriesError,
    TimeSeriesFrame,
    prepare_forecast_data,
    read_csv_series,
    synthetic_financial_series,
)
from crucible.forecast.metrics import (
    apply_quantile_calibration,
    evaluate_forecasts,
    fit_quantile_calibration,
    moving_block_bootstrap_skill,
)
from crucible.forecast.model import PatchTSTConfig, build_patchtst, quantile_loss
from crucible.forecast.pipeline import ForecastRunConfig, ForecastRunResult, run_forecast
from crucible.forecast.trainer import (
    ForecastTrainConfig,
    load_forecaster,
    predict,
    train_forecaster,
)

__all__ = [
    "ForecastDataBundle",
    "ForecastRunConfig",
    "ForecastRunResult",
    "ForecastTrainConfig",
    "PatchTSTConfig",
    "SyntheticFinancialConfig",
    "TimeSeriesError",
    "TimeSeriesFrame",
    "apply_quantile_calibration",
    "build_patchtst",
    "evaluate_forecasts",
    "fit_quantile_calibration",
    "load_forecaster",
    "moving_block_bootstrap_skill",
    "predict",
    "prepare_forecast_data",
    "quantile_loss",
    "read_csv_series",
    "run_forecast",
    "synthetic_financial_series",
    "train_forecaster",
]
