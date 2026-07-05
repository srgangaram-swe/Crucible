"""Distribution drift between datasets via the Population Stability Index.

PSI (a symmetrized KL-divergence estimate over binned distributions) with
the standard banking-industry thresholds: < 0.1 no drift, 0.1–0.25 moderate,
> 0.25 major (Siddiqi, *Credit Risk Scorecards*, 2006). We profile two
things a text corpus can drift on without any schema change: the source mix
and the document-length distribution.

Profiles are plain JSON artifacts, so a reference profile can be committed
next to a dataset version and compared against any later ingest.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

# Fixed log-spaced word-count bin edges so profiles are comparable across
# runs; the last bin is open-ended.
_LENGTH_BIN_EDGES = (0, 5, 10, 20, 40, 80, 160, 320, 640)

_EPSILON = 1e-4  # smoothing so empty bins do not blow up the log-ratio


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    n_rows: int
    source_dist: dict[str, float]
    length_dist: list[float]  # over _LENGTH_BIN_EDGES bins

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> DatasetProfile:
        raw = json.loads(path.read_text())
        return cls(
            n_rows=raw["n_rows"],
            source_dist=raw["source_dist"],
            length_dist=raw["length_dist"],
        )


def _length_bin(n_words: int) -> int:
    for i in range(len(_LENGTH_BIN_EDGES) - 1, -1, -1):
        if n_words >= _LENGTH_BIN_EDGES[i]:
            return i
    return 0


def profile_table(
    table: pa.Table, text_column: str = "text", source_column: str = "source"
) -> DatasetProfile:
    n = table.num_rows
    if n == 0:
        raise ValueError("cannot profile an empty table")

    source_counts: dict[str, int] = {}
    for value in table.column(source_column).to_pylist():
        source_counts[str(value)] = source_counts.get(str(value), 0) + 1

    length_counts = [0] * len(_LENGTH_BIN_EDGES)
    for text in table.column(text_column).to_pylist():
        words = len(text.split()) if text else 0
        length_counts[_length_bin(words)] += 1

    return DatasetProfile(
        n_rows=n,
        source_dist={k: v / n for k, v in sorted(source_counts.items())},
        length_dist=[c / n for c in length_counts],
    )


def population_stability_index(expected: list[float], actual: list[float]) -> float:
    if len(expected) != len(actual):
        raise ValueError("distributions must share binning")
    psi = 0.0
    for e, a in zip(expected, actual, strict=True):
        e_s, a_s = max(e, _EPSILON), max(a, _EPSILON)
        psi += (a_s - e_s) * math.log(a_s / e_s)
    return psi


def _aligned_categorical(
    reference: dict[str, float], current: dict[str, float]
) -> tuple[list[float], list[float]]:
    keys = sorted(set(reference) | set(current))
    return [reference.get(k, 0.0) for k in keys], [current.get(k, 0.0) for k in keys]


def _verdict(psi: float) -> str:
    if psi < 0.1:
        return "none"
    if psi <= 0.25:
        return "moderate"
    return "major"


def drift_report(reference: DatasetProfile, current: DatasetProfile) -> dict[str, Any]:
    ref_sources, cur_sources = _aligned_categorical(reference.source_dist, current.source_dist)
    source_psi = population_stability_index(ref_sources, cur_sources)
    length_psi = population_stability_index(reference.length_dist, current.length_dist)
    worst = max(source_psi, length_psi)
    return {
        "source_psi": round(source_psi, 4),
        "length_psi": round(length_psi, 4),
        "source_verdict": _verdict(source_psi),
        "length_verdict": _verdict(length_psi),
        "verdict": _verdict(worst),
        "reference_rows": reference.n_rows,
        "current_rows": current.n_rows,
    }
