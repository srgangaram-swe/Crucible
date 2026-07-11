"""Dependency-free SVG evaluation plots suitable for committed artifacts."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np


def _polyline(
    values: np.ndarray,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    bounds: tuple[float, float] | None = None,
) -> str:
    if not len(values):
        return ""
    low, high = bounds or (float(np.min(values)), float(np.max(values)))
    span = high - low or 1.0
    points = [
        f"{x0 + width * i / max(1, len(values) - 1):.1f},{y0 + height * (high - float(v)) / span:.1f}"
        for i, v in enumerate(values)
    ]
    return " ".join(points)


def _write(path: Path, title: str, body: str, subtitle: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="460" viewBox="0 0 900 460">'
        f"<title>{html.escape(title)}</title><desc>{html.escape(subtitle)}</desc>"
        '<rect width="100%" height="100%" fill="#fbfcfe"/>'
        f'<text x="36" y="36" font-family="sans-serif" font-size="22" fill="#18212f">{html.escape(title)}</text>'
        f'<text x="36" y="58" font-family="sans-serif" font-size="12" fill="#5b6675">{html.escape(subtitle)}</text>'
        f"{body}</svg>\n"
    )
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    return path


def forecast_plot(
    path: Path,
    actual: np.ndarray,
    predictions: np.ndarray,
    quantiles: tuple[float, ...],
    *,
    max_points: int = 180,
) -> Path:
    y = actual[:max_points, 0]
    median = predictions[:max_points, 0, quantiles.index(0.5)]
    low = predictions[:max_points, 0, 0]
    high = predictions[:max_points, 0, -1]
    all_values = np.concatenate([y, median, low, high])
    minimum, maximum = float(all_values.min()), float(all_values.max())
    span = maximum - minimum or 1.0

    def points(values: np.ndarray) -> str:
        return " ".join(
            f"{70 + 780 * i / max(1, len(values) - 1):.1f},{405 - 310 * (float(value) - minimum) / span:.1f}"
            for i, value in enumerate(values)
        )

    band = f'<polygon points="{points(low)} {" ".join(reversed(points(high).split()))}" fill="#8ebce6" opacity="0.28"/>'
    body = (
        '<line x1="70" y1="405" x2="850" y2="405" stroke="#9aa4b2"/>'
        + band
        + f'<polyline points="{points(y)}" fill="none" stroke="#222b38" stroke-width="1.6"/>'
        + f'<polyline points="{points(median)}" fill="none" stroke="#1565a7" stroke-width="1.8"/>'
        + '<text x="70" y="438" font-family="sans-serif" font-size="12">actual (dark), median forecast (blue), prediction interval (band)</text>'
    )
    return _write(path, "Out-of-sample forecast", body, "First test horizon; chronology preserved")


def residual_plot(path: Path, actual: np.ndarray, median: np.ndarray, bins: int = 24) -> Path:
    residuals = (actual - median).ravel()
    histogram_range = None
    if float(np.ptp(residuals)) < 1e-12:
        center = float(residuals[0])
        histogram_range = (center - 1e-6, center + 1e-6)
    counts, _ = np.histogram(residuals, bins=bins, range=histogram_range)
    maximum = max(1, int(counts.max()))
    bar_width = 760 / bins
    bars = "".join(
        f'<rect x="{75 + i * bar_width:.1f}" y="{405 - 300 * count / maximum:.1f}" width="{bar_width - 2:.1f}" height="{300 * count / maximum:.1f}" fill="#4c7c9f"/>'
        for i, count in enumerate(counts)
    )
    return _write(
        path,
        "Residual distribution",
        bars,
        f"n={len(residuals)}; mean={float(residuals.mean()):.6f}",
    )


def training_plot(path: Path, history: list[dict[str, float]]) -> Path:
    train = np.array([row["train_loss"] for row in history])
    validation = np.array([row["validation_loss"] for row in history])
    bounds = (
        float(min(train.min(), validation.min())),
        float(max(train.max(), validation.max())),
    )
    body = (
        f'<polyline points="{_polyline(train, x0=70, y0=90, width=760, height=300, bounds=bounds)}" fill="none" stroke="#a65a2e" stroke-width="2"/>'
        f'<polyline points="{_polyline(validation, x0=70, y0=90, width=760, height=300, bounds=bounds)}" fill="none" stroke="#1565a7" stroke-width="2"/>'
        '<text x="70" y="430" font-family="sans-serif" font-size="12">training (orange), validation (blue)</text>'
    )
    return _write(path, "Optimization history", body, "Pinball loss by epoch")


def calibration_plot(
    path: Path, actual: np.ndarray, predictions: np.ndarray, quantiles: tuple[float, ...]
) -> Path:
    empirical = np.array(
        [float(np.mean(actual <= predictions[:, :, i])) for i in range(len(quantiles))]
    )
    nominal = np.array(quantiles)
    points = " ".join(
        f"{100 + 700 * q:.1f},{405 - 310 * observed:.1f}"
        for q, observed in zip(nominal, empirical, strict=True)
    )
    body = (
        '<line x1="100" y1="405" x2="800" y2="95" stroke="#9aa4b2" stroke-dasharray="6 5"/>'
        f'<polyline points="{points}" fill="none" stroke="#6a3d9a" stroke-width="2"/>'
        + "".join(
            f'<circle cx="{100 + 700 * q:.1f}" cy="{405 - 310 * observed:.1f}" r="5" fill="#6a3d9a"/>'
            for q, observed in zip(nominal, empirical, strict=True)
        )
    )
    return _write(path, "Quantile calibration", body, "Dashed line is ideal calibration")


def horizon_metrics_plot(
    path: Path,
    actual: np.ndarray,
    median: np.ndarray,
    baseline: np.ndarray,
) -> Path:
    model_mae = np.mean(np.abs(actual - median), axis=0)
    baseline_mae = np.mean(np.abs(actual - baseline), axis=0)
    maximum = max(float(model_mae.max()), float(baseline_mae.max()), 1e-12)
    group_width = 700 / len(model_mae)
    bars: list[str] = []
    for index, (model_value, baseline_value) in enumerate(
        zip(model_mae, baseline_mae, strict=True)
    ):
        x = 95 + index * group_width
        model_height = 280 * float(model_value) / maximum
        baseline_height = 280 * float(baseline_value) / maximum
        bars.extend(
            [
                f'<rect x="{x:.1f}" y="{405 - model_height:.1f}" width="{group_width * 0.34:.1f}" height="{model_height:.1f}" fill="#1565a7"/>',
                f'<rect x="{x + group_width * 0.4:.1f}" y="{405 - baseline_height:.1f}" width="{group_width * 0.34:.1f}" height="{baseline_height:.1f}" fill="#a65a2e"/>',
                f'<text x="{x + group_width * 0.2:.1f}" y="430" font-family="sans-serif" font-size="11">h{index + 1}</text>',
            ]
        )
    body = "".join(bars) + (
        '<text x="70" y="455" font-family="sans-serif" font-size="12">MAE: model (blue), persistence (orange)</text>'
    )
    return _write(path, "Error by forecast horizon", body, "Held-out test MAE")


def write_evaluation_plots(
    directory: Path,
    *,
    actual: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
    quantiles: tuple[float, ...],
    history: list[dict[str, float]],
) -> dict[str, str]:
    median = predictions[:, :, quantiles.index(0.5)]
    outputs = {
        "forecast": forecast_plot(directory / "forecast.svg", actual, predictions, quantiles),
        "residuals": residual_plot(directory / "residuals.svg", actual, median),
        "training": training_plot(directory / "training.svg", history),
        "calibration": calibration_plot(
            directory / "calibration.svg", actual, predictions, quantiles
        ),
        "metrics_by_horizon": horizon_metrics_plot(
            directory / "metrics_by_horizon.svg", actual, median, baseline
        ),
    }
    return {name: str(path) for name, path in outputs.items()}
