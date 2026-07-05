"""The bronze→silver promotion gate with a quarantine path.

Promotion is a pure function of (bronze content, config): rerunning it
rebuilds silver/<dataset> and quarantine/<dataset> from scratch (derived
layers are always rebuildable, so clearing them is safe). Records failing
any enabled rule land in quarantine with a ``reject_reasons`` column; in a
production system PII rows would be redacted rather than parked, which is
called out in docs/limitations.md.

If more than ``max_reject_rate`` of records fail, nothing is promoted and
the verdict is ``blocked``: a systematically broken source must fail loudly
instead of leaking a plausible-looking silver dataset.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, Field, field_validator

from crucible.ingest.land import batch_hash
from crucible.quality.rules import RULES, evaluate_text
from crucible.storage import Catalog, Layer


class QualityConfig(BaseModel):
    """Gate policy: which rules run and how strict promotion is."""

    # no_repeated_sentences exists in the registry but is opt-in; see its
    # docstring for the measured reason.
    rules: list[str] = Field(
        default=[
            "non_empty",
            "min_words",
            "no_mojibake",
            "no_boilerplate_markers",
            "no_pii",
        ]
    )
    text_column: str = "text"
    id_column: str = "id"
    min_words: int = Field(default=5, ge=1)
    boilerplate_marker_threshold: int = Field(default=2, ge=1)
    max_duplicate_sentence_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    max_reject_rate: float = Field(default=0.5, gt=0.0, le=1.0)
    part_rows: int = Field(default=1000, ge=1)

    @field_validator("rules")
    @classmethod
    def _known_rules(cls, rules: list[str]) -> list[str]:
        unknown = set(rules) - set(RULES)
        if unknown:
            raise ValueError(f"unknown rules {sorted(unknown)}; known: {sorted(RULES)}")
        return rules


@dataclass(frozen=True, slots=True)
class GateResult:
    dataset: str
    verdict: str  # "promoted" | "blocked"
    input_rows: int
    promoted_rows: int
    quarantined_rows: int
    reject_rate: float
    reasons_histogram: dict[str, int]
    duplicate_id_rows: int  # informational; duplicates are dedup's job
    elapsed_s: float
    silver_parts: int = 0
    quarantine_parts: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clear_dataset(catalog: Catalog, layer: Layer, dataset: str) -> None:
    directory = catalog.dataset_dir(layer, dataset)
    if directory.exists():
        shutil.rmtree(directory)


def _write_parts(catalog: Catalog, layer: Layer, dataset: str, table: pa.Table, rows: int) -> int:
    parts = 0
    for start in range(0, table.num_rows, rows):
        chunk = table.slice(start, rows)
        catalog.write_part(chunk, layer, dataset, f"part-{batch_hash(chunk)[:16]}")
        parts += 1
    return parts


def run_gate(
    catalog: Catalog,
    dataset: str,
    cfg: QualityConfig,
    report_dir: Path | None = None,
) -> GateResult:
    """Validate bronze/<dataset> and promote survivors to silver."""
    started = time.perf_counter()
    table = catalog.read(Layer.BRONZE, dataset)
    if cfg.text_column not in table.column_names:
        raise ValueError(f"bronze/{dataset} has no {cfg.text_column!r} column")

    texts: list[str] = [
        value if value is not None else "" for value in table.column(cfg.text_column).to_pylist()
    ]
    reasons_per_row = [evaluate_text(text, cfg) for text in texts]

    histogram: dict[str, int] = {}
    for reasons in reasons_per_row:
        for reason in reasons:
            histogram[reason] = histogram.get(reason, 0) + 1

    quarantine_mask = [bool(reasons) for reasons in reasons_per_row]
    quarantined = sum(quarantine_mask)
    reject_rate = quarantined / table.num_rows

    duplicate_id_rows = 0
    if cfg.id_column in table.column_names:
        ids = table.column(cfg.id_column).to_pylist()
        duplicate_id_rows = len(ids) - len(set(ids))

    notes: list[str] = []
    if duplicate_id_rows:
        notes.append(
            f"{duplicate_id_rows} duplicate-id rows promoted as-is; removal is dedup's job"
        )

    if reject_rate > cfg.max_reject_rate:
        return GateResult(
            dataset=dataset,
            verdict="blocked",
            input_rows=table.num_rows,
            promoted_rows=0,
            quarantined_rows=quarantined,
            reject_rate=round(reject_rate, 4),
            reasons_histogram=dict(sorted(histogram.items())),
            duplicate_id_rows=duplicate_id_rows,
            elapsed_s=round(time.perf_counter() - started, 3),
            notes=[*notes, f"reject rate {reject_rate:.1%} exceeds max {cfg.max_reject_rate:.1%}"],
        )

    promote_indices = [i for i, bad in enumerate(quarantine_mask) if not bad]
    quarantine_indices = [i for i, bad in enumerate(quarantine_mask) if bad]

    _clear_dataset(catalog, Layer.SILVER, dataset)
    _clear_dataset(catalog, Layer.QUARANTINE, dataset)

    silver_parts = 0
    if promote_indices:
        silver_table = table.take(promote_indices)
        silver_parts = _write_parts(catalog, Layer.SILVER, dataset, silver_table, cfg.part_rows)

    quarantine_parts = 0
    if quarantine_indices:
        quarantine_table = table.take(quarantine_indices).append_column(
            "reject_reasons",
            pa.array(["|".join(reasons_per_row[i]) for i in quarantine_indices]),
        )
        quarantine_parts = _write_parts(
            catalog, Layer.QUARANTINE, dataset, quarantine_table, cfg.part_rows
        )

    return GateResult(
        dataset=dataset,
        verdict="promoted",
        input_rows=table.num_rows,
        promoted_rows=len(promote_indices),
        quarantined_rows=quarantined,
        reject_rate=round(reject_rate, 4),
        reasons_histogram=dict(sorted(histogram.items())),
        duplicate_id_rows=duplicate_id_rows,
        elapsed_s=round(time.perf_counter() - started, 3),
        silver_parts=silver_parts,
        quarantine_parts=quarantine_parts,
        notes=notes,
    )
