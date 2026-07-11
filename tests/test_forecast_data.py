import csv
from pathlib import Path

import numpy as np
import pytest

from crucible.forecast.data import (
    StandardScaler,
    SyntheticFinancialConfig,
    TimeSeriesError,
    TimeSeriesFrame,
    chronological_split,
    prepare_forecast_data,
    read_csv_series,
    synthetic_financial_series,
)


def test_frame_rejects_bad_shapes_values_and_time() -> None:
    timestamps = np.arange(4).astype("datetime64[D]")
    values = np.ones((4, 1), dtype=np.float32)
    with pytest.raises(TimeSeriesError, match="strictly increasing"):
        TimeSeriesFrame(timestamps[::-1], values, ("y",), "y")
    corrupt = values.copy()
    corrupt[2, 0] = np.nan
    with pytest.raises(TimeSeriesError, match="finite"):
        TimeSeriesFrame(timestamps, corrupt, ("y",), "y")
    with pytest.raises(TimeSeriesError, match="not a feature"):
        TimeSeriesFrame(timestamps, values, ("x",), "y")
    with pytest.raises(TimeSeriesError, match="lengths differ"):
        TimeSeriesFrame(timestamps[:-1], values, ("y",), "y")
    with pytest.raises(TimeSeriesError, match="unique"):
        TimeSeriesFrame(timestamps, np.ones((4, 2)), ("y", "y"), "y")
    with pytest.raises(TimeSeriesError, match="three rows"):
        TimeSeriesFrame(timestamps[:2], values[:2], ("y",), "y")


def test_csv_adapter_round_trip_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "return", "volume"])
        for day in range(5):
            writer.writerow([f"2026-01-{day + 1:02d}", day / 100, 1000 + day])
    frame = read_csv_series(
        path,
        timestamp_column="timestamp",
        target_column="return",
        feature_columns=["return", "volume"],
    )
    assert frame.columns == ("return", "volume")
    assert frame.values.shape == (5, 2)
    with pytest.raises(TimeSeriesError, match="missing column"):
        read_csv_series(path, timestamp_column="date", target_column="return")
    empty = tmp_path / "empty.csv"
    empty.write_text("timestamp,return\n")
    with pytest.raises(TimeSeriesError, match="no rows"):
        read_csv_series(empty, timestamp_column="timestamp", target_column="return")
    path.write_text("timestamp,return\nnot-a-date,bad\n")
    with pytest.raises(TimeSeriesError, match="invalid timestamp"):
        read_csv_series(path, timestamp_column="timestamp", target_column="return")


def test_chronological_windows_are_sealed_and_embargoed() -> None:
    frame = synthetic_financial_series(SyntheticFinancialConfig(n_steps=320, seed=7))
    bundle = prepare_forecast_data(frame, context_length=32, horizon=5, embargo=5)
    assert bundle.split.validation_start - bundle.split.train_end == 5
    assert bundle.split.test_start - bundle.split.validation_end == 5
    assert bundle.train.origins[-1] + 5 <= bundle.split.train_end
    assert bundle.validation.origins[0] >= bundle.split.validation_start
    assert bundle.validation.origins[-1] + 5 <= bundle.split.validation_end
    assert bundle.test.origins[0] >= bundle.split.test_start
    assert set(bundle.train.forecast_timestamps.ravel()).isdisjoint(
        set(bundle.validation.forecast_timestamps.ravel())
    )
    assert set(bundle.validation.forecast_timestamps.ravel()).isdisjoint(
        set(bundle.test.forecast_timestamps.ravel())
    )


def test_scaler_is_fit_on_train_only_and_future_sentinel_cannot_change_it() -> None:
    frame = synthetic_financial_series(SyntheticFinancialConfig(n_steps=300, seed=8))
    first = prepare_forecast_data(frame, context_length=24, horizon=3, embargo=3)
    changed = frame.values.copy()
    changed[first.split.test_start :] = 1_000_000
    second = prepare_forecast_data(
        TimeSeriesFrame(frame.timestamps, changed, frame.columns, frame.target),
        context_length=24,
        horizon=3,
        embargo=3,
    )
    np.testing.assert_array_equal(first.scaler.mean, second.scaler.mean)
    np.testing.assert_array_equal(first.scaler.scale, second.scaler.scale)
    np.testing.assert_array_equal(first.train.inputs, second.train.inputs)
    np.testing.assert_array_equal(first.validation.inputs, second.validation.inputs)


def test_split_and_scaler_validation() -> None:
    with pytest.raises(TimeSeriesError, match="invalid row count"):
        chronological_split(10)
    with pytest.raises(TimeSeriesError, match="empty split"):
        chronological_split(100, train_fraction=0.7, validation_fraction=0.2, embargo=30)
    with pytest.raises(TimeSeriesError, match="boundary"):
        StandardScaler.fit(np.ones((4, 2)), 0)
    frame = synthetic_financial_series(SyntheticFinancialConfig(n_steps=220))
    with pytest.raises(TimeSeriesError, match="embargo"):
        prepare_forecast_data(frame, context_length=24, horizon=5, embargo=2)
    with pytest.raises(TimeSeriesError, match="too short"):
        prepare_forecast_data(frame, context_length=190, horizon=10, embargo=10)
