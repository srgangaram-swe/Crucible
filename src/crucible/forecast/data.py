"""Leakage-safe time-series contracts, splits, windows, and synthetic finance data."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class TimeSeriesError(ValueError):
    """Invalid temporal data or a split/window request that could leak."""


@dataclass(frozen=True, slots=True)
class TimeSeriesFrame:
    timestamps: np.ndarray
    values: np.ndarray
    columns: tuple[str, ...]
    target: str

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise TimeSeriesError("values must have shape [time, features]")
        if len(self.timestamps) != len(self.values):
            raise TimeSeriesError("timestamp and value lengths differ")
        if self.values.shape[1] != len(self.columns) or len(set(self.columns)) != len(self.columns):
            raise TimeSeriesError("columns must be unique and match the feature dimension")
        if self.target not in self.columns:
            raise TimeSeriesError(f"target {self.target!r} is not a feature column")
        if len(self.timestamps) < 3:
            raise TimeSeriesError("time series requires at least three rows")
        if not np.all(np.isfinite(self.values)):
            raise TimeSeriesError("time-series values must be finite")
        stamps = self.timestamps.astype("datetime64[ns]").astype(np.int64)
        if np.any(np.diff(stamps) <= 0):
            raise TimeSeriesError("timestamps must be unique and strictly increasing")

    @property
    def target_index(self) -> int:
        return self.columns.index(self.target)


class SyntheticFinancialConfig(BaseModel):
    """Regime-switching return process with predictive lag/volatility covariates."""

    n_steps: int = Field(default=1800, ge=200)
    seed: int = 41
    start: datetime = datetime(2018, 1, 1, tzinfo=UTC)
    frequency_minutes: int = Field(default=1440, ge=1)
    regime_persistence: float = Field(default=0.97, gt=0.5, lt=1.0)
    low_volatility: float = Field(default=0.006, gt=0)
    high_volatility: float = Field(default=0.022, gt=0)


def synthetic_financial_series(config: SyntheticFinancialConfig) -> TimeSeriesFrame:
    """Generate realistic-enough returns for offline plumbing tests, not market simulation."""
    rng = random.Random(config.seed)
    returns: list[float] = []
    volumes: list[float] = []
    realized_volatility: list[float] = []
    momentum: list[float] = []
    regime = 0
    latent_momentum = 0.0
    log_volume = 12.0
    for step in range(config.n_steps):
        if rng.random() > config.regime_persistence:
            regime = 1 - regime
        volatility = config.high_volatility if regime else config.low_volatility
        weekly = math.sin(2 * math.pi * step / 5) * 0.0008
        shock = rng.gauss(0.0, volatility)
        value = 0.28 * latent_momentum + weekly + shock
        latent_momentum = 0.92 * latent_momentum + 0.08 * value
        log_volume = 0.94 * log_volume + 0.06 * (12.0 + 18.0 * abs(value)) + rng.gauss(0, 0.04)
        returns.append(value)
        volumes.append(log_volume)
        window = returns[-20:]
        realized_volatility.append(float(np.std(window)) if len(window) > 1 else volatility)
        momentum.append(float(np.mean(returns[-10:])))
    timestamps = np.array(
        [
            np.datetime64(
                (config.start + timedelta(minutes=config.frequency_minutes * step)).replace(
                    tzinfo=None
                ),
                "ns",
            )
            for step in range(config.n_steps)
        ]
    )
    values = np.column_stack([returns, realized_volatility, momentum, volumes]).astype(np.float32)
    return TimeSeriesFrame(
        timestamps=timestamps,
        values=values,
        columns=("return", "realized_volatility", "momentum_10", "log_volume"),
        target="return",
    )


def read_csv_series(
    path: Path,
    *,
    timestamp_column: str,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> TimeSeriesFrame:
    """Read timestamped numeric columns without imposing a finance vendor schema."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise TimeSeriesError(f"{path}: CSV contains no rows")
    columns = tuple(feature_columns or [target_column])
    if target_column not in columns:
        columns = (target_column, *columns)
    try:
        parsed_timestamps = []
        for row in rows:
            parsed = datetime.fromisoformat(row[timestamp_column].replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            parsed_timestamps.append(
                np.datetime64(parsed.astimezone(UTC).replace(tzinfo=None), "ns")
            )
        timestamps = np.array(parsed_timestamps)
        values = np.array(
            [[float(row[column]) for column in columns] for row in rows], dtype=np.float32
        )
    except KeyError as exc:
        raise TimeSeriesError(f"{path}: missing column {exc.args[0]!r}") from exc
    except ValueError as exc:
        raise TimeSeriesError(f"{path}: invalid timestamp or numeric value: {exc}") from exc
    return TimeSeriesFrame(timestamps, values, columns, target_column)


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    n_rows: int
    embargo: int

    def as_dict(self) -> dict[str, int]:
        return {
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "n_rows": self.n_rows,
            "embargo": self.embargo,
        }


def chronological_split(
    n_rows: int,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    embargo: int = 0,
) -> SplitBoundaries:
    if n_rows < 20 or not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise TimeSeriesError("invalid row count or split fractions")
    train_end = int(n_rows * train_fraction)
    validation_start = train_end + embargo
    validation_end = validation_start + int(n_rows * validation_fraction)
    test_start = validation_end + embargo
    if min(train_end, validation_end - validation_start, n_rows - test_start) < 1:
        raise TimeSeriesError("embargo and fractions leave an empty split")
    return SplitBoundaries(train_end, validation_start, validation_end, test_start, n_rows, embargo)


@dataclass(frozen=True, slots=True)
class StandardScaler:
    mean: np.ndarray
    scale: np.ndarray
    fitted_through: int

    @classmethod
    def fit(cls, values: np.ndarray, fitted_through: int) -> StandardScaler:
        if fitted_through <= 0 or fitted_through > len(values):
            raise TimeSeriesError("scaler boundary is outside the series")
        training = values[:fitted_through]
        scale = training.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(training.mean(axis=0), scale, fitted_through)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray((values - self.mean) / self.scale, dtype=np.float32)

    def inverse_target(self, values: np.ndarray, target_index: int) -> np.ndarray:
        return np.asarray(
            values * self.scale[target_index] + self.mean[target_index], dtype=np.float32
        )


@dataclass(frozen=True, slots=True)
class WindowSet:
    inputs: np.ndarray
    targets: np.ndarray
    forecast_timestamps: np.ndarray
    origins: np.ndarray

    def __post_init__(self) -> None:
        if self.inputs.ndim != 3 or self.targets.ndim != 2:
            raise TimeSeriesError(
                "windows must be [sample, context, feature] and [sample, horizon]"
            )
        if len(self.inputs) != len(self.targets) or len(self.origins) != len(self.targets):
            raise TimeSeriesError("window sample counts differ")


def make_windows(
    frame: TimeSeriesFrame,
    scaled_values: np.ndarray,
    *,
    context_length: int,
    horizon: int,
    target_start: int,
    target_end: int,
) -> WindowSet:
    if context_length < 2 or horizon < 1 or scaled_values.shape != frame.values.shape:
        raise TimeSeriesError("invalid context, horizon, or scaled values")
    first_origin = max(context_length, target_start)
    final_origin = target_end - horizon
    if first_origin > final_origin:
        raise TimeSeriesError("split is too short for requested context and horizon")
    origins = np.arange(first_origin, final_origin + 1)
    inputs = np.stack([scaled_values[i - context_length : i] for i in origins])
    targets = np.stack([scaled_values[i : i + horizon, frame.target_index] for i in origins])
    stamps = np.stack([frame.timestamps[i : i + horizon] for i in origins])
    return WindowSet(inputs.astype(np.float32), targets.astype(np.float32), stamps, origins)


@dataclass(frozen=True, slots=True)
class ForecastDataBundle:
    train: WindowSet
    validation: WindowSet
    test: WindowSet
    scaler: StandardScaler
    split: SplitBoundaries
    target_index: int
    columns: tuple[str, ...]
    training_target_original: np.ndarray


def prepare_forecast_data(
    frame: TimeSeriesFrame,
    *,
    context_length: int,
    horizon: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    embargo: int | None = None,
) -> ForecastDataBundle:
    gap = horizon if embargo is None else embargo
    split = chronological_split(
        len(frame.values),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        embargo=gap,
    )
    if gap < horizon:
        raise TimeSeriesError("embargo must be at least the forecast horizon")
    scaler = StandardScaler.fit(frame.values, split.train_end)
    scaled = scaler.transform(frame.values)
    kwargs: dict[str, Any] = {
        "frame": frame,
        "scaled_values": scaled,
        "context_length": context_length,
        "horizon": horizon,
    }
    return ForecastDataBundle(
        train=make_windows(**kwargs, target_start=context_length, target_end=split.train_end),
        validation=make_windows(
            **kwargs, target_start=split.validation_start, target_end=split.validation_end
        ),
        test=make_windows(**kwargs, target_start=split.test_start, target_end=split.n_rows),
        scaler=scaler,
        split=split,
        target_index=frame.target_index,
        columns=frame.columns,
        training_target_original=frame.values[: split.train_end, frame.target_index].copy(),
    )
