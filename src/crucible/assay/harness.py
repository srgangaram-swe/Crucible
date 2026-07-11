"""Content-addressed, multi-seed experiment execution and reporting."""

from __future__ import annotations

import csv
import io
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from crucible import __version__
from crucible.utils.hashing import canonical_json, sha256_texts


class ExperimentConfig(BaseModel):
    """Reproducible study configuration shared by all Phase 8 experiments."""

    study: str
    seeds: list[int] = Field(default_factory=lambda: [11, 23, 37], min_length=1)
    n_docs: int = Field(default=240, ge=40)
    train_tokens: int = Field(default=20_000, ge=1_000)
    bootstrap_samples: int = Field(default=1_000, ge=100)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        return sha256_texts(
            [canonical_json(self.model_dump(mode="json")), f"crucible/{__version__}"]
        )[:12]


@dataclass(frozen=True)
class ExperimentResult:
    study: str
    config_hash: str
    result_hash: str
    artifact_dir: Path
    rows: list[dict[str, Any]]
    summary: list[dict[str, Any]]


Study = Callable[[ExperimentConfig, int], list[dict[str, Any]]]


def bootstrap_ci(
    values: list[float], *, samples: int, seed: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, deterministic for ``seed``."""
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    tail = (1 - confidence) / 2
    return means[int(tail * samples)], means[min(samples - 1, int((1 - tail) * samples))]


def _summarize(config: ExperimentConfig, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = sorted(
        key
        for key, value in rows[0].items()
        if key not in {"seed", "arm"} and isinstance(value, int | float)
    )
    summary: list[dict[str, Any]] = []
    for arm in sorted({str(row["arm"]) for row in rows}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        for metric in metrics:
            values = [float(row[metric]) for row in arm_rows if metric in row]
            if not values:
                continue
            low, high = bootstrap_ci(
                values,
                samples=config.bootstrap_samples,
                seed=int(config.config_hash[:8], 16) ^ sum(ord(c) for c in arm + metric),
            )
            summary.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "mean": round(float(np.mean(values)), 6),
                    "ci_low": round(low, 6),
                    "ci_high": round(high, 6),
                    "n_seeds": len(values),
                }
            )
    return summary


def _csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    fields = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _markdown(config: ExperimentConfig, summary: list[dict[str, Any]]) -> str:
    lines = [
        f"# {config.study} results",
        "",
        f"Config hash: `{config.config_hash}`. Seeds: `{config.seeds}`.",
        "",
        "| arm | metric | mean | 95% bootstrap CI | n |",
        "|---|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['arm']} | {row['metric']} | {row['mean']} | "
        f"[{row['ci_low']}, {row['ci_high']}] | {row['n_seeds']} |"
        for row in summary
    )
    return "\n".join(lines) + "\n"


def _svg(config: ExperimentConfig, summary: list[dict[str, Any]]) -> str:
    primary = [row for row in summary if row["metric"] == "validation_loss"]
    width, height = 720, 360
    if not primary:
        primary = summary[:8]
    maximum = max((float(row["ci_high"]) for row in primary), default=1.0) or 1.0
    bars: list[str] = []
    bar_width = max(20, 560 // max(1, len(primary)))
    for index, row in enumerate(primary):
        x = 100 + index * bar_width
        bar_height = 230 * float(row["mean"]) / maximum
        y = 290 - bar_height
        low_y = 290 - 230 * float(row["ci_low"]) / maximum
        high_y = 290 - 230 * float(row["ci_high"]) / maximum
        center = x + (bar_width - 8) / 2
        bars.extend(
            [
                f'<rect x="{x}" y="{y:.1f}" width="{bar_width - 8}" height="{bar_height:.1f}" fill="#35618f"/>',
                f'<line x1="{center:.1f}" y1="{high_y:.1f}" x2="{center:.1f}" y2="{low_y:.1f}" stroke="#111"/>',
                f'<line x1="{center - 5:.1f}" y1="{high_y:.1f}" x2="{center + 5:.1f}" y2="{high_y:.1f}" stroke="#111"/>',
                f'<line x1="{center - 5:.1f}" y1="{low_y:.1f}" x2="{center + 5:.1f}" y2="{low_y:.1f}" stroke="#111"/>',
                f'<text x="{x}" y="315" font-size="11" transform="rotate(25 {x} 315)">{row["arm"]}</text>',
            ]
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        f'<text x="20" y="28" font-size="18">{config.study}: validation loss</text>'
        f'<text x="20" y="48" font-size="11">config {config.config_hash}; error bars are bootstrap 95% CIs</text>'
        + "".join(bars)
        + "</svg>\n"
    )


def run_experiment(config: ExperimentConfig, study: Study, output_root: Path) -> ExperimentResult:
    """Execute every seed and atomically publish JSON/CSV/Markdown/SVG artifacts."""
    rows = [row for seed in config.seeds for row in study(config, seed)]
    if not rows or any("arm" not in row or "seed" not in row for row in rows):
        raise ValueError("study must return at least one row with arm and seed")
    summary = _summarize(config, rows)
    result_hash = sha256_texts([config.config_hash, canonical_json(rows)])[:12]
    artifact_dir = Path(output_root) / config.study / config.config_hash / result_hash
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "study": config.study,
        "config_hash": config.config_hash,
        "result_hash": result_hash,
        "code_version": __version__,
        "config": config.model_dump(mode="json"),
        "rows": rows,
        "summary": summary,
    }
    artifacts = {
        "results.json": json.dumps(payload, indent=2, sort_keys=True) + "\n",
        "results.csv": _csv(rows),
        "report.md": _markdown(config, summary),
        "plot.svg": _svg(config, summary),
    }
    for name, content in artifacts.items():
        target = artifact_dir / name
        tmp = artifact_dir / f".{name}.tmp"
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(target)
    return ExperimentResult(
        config.study, config.config_hash, result_hash, artifact_dir, rows, summary
    )
