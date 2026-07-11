"""Durable, dependency-free stage metrics for local pipeline operations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class StageMetric:
    run_id: str
    stage: str
    status: str
    started_at: str
    duration_seconds: float
    input_rows: int | None = None
    output_rows: int | None = None
    attempt: int = 1
    error: str | None = None

    @property
    def throughput_rows_per_second(self) -> float | None:
        if self.output_rows is None or self.duration_seconds <= 0:
            return None
        return self.output_rows / self.duration_seconds

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["throughput_rows_per_second"] = self.throughput_rows_per_second
        return payload


class MetricsStore:
    """Append-only JSONL metric store; malformed trailing records are ignored."""

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "metrics" / "stages.jsonl"

    def append(self, metric: StageMetric) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric.as_dict(), sort_keys=True) + "\n")

    def list(self, *, run_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists() or limit <= 0:
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run_id is None or record.get("run_id") == run_id:
                records.append(record)
        return records[-limit:]


class StageTimer:
    """Context manager that emits a metric on both success and failure."""

    def __init__(self, store: MetricsStore, run_id: str, stage: str, *, attempt: int = 1) -> None:
        self.store, self.run_id, self.stage, self.attempt = store, run_id, stage, attempt
        self.input_rows: int | None = None
        self.output_rows: int | None = None
        self._started_at = ""
        self._start = 0.0

    def __enter__(self) -> StageTimer:
        self._started_at = datetime.now(UTC).isoformat()
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.store.append(
            StageMetric(
                run_id=self.run_id,
                stage=self.stage,
                status="failed" if exc is not None else "complete",
                started_at=self._started_at,
                duration_seconds=perf_counter() - self._start,
                input_rows=self.input_rows,
                output_rows=self.output_rows,
                attempt=self.attempt,
                error=str(exc) if exc is not None else None,
            )
        )
