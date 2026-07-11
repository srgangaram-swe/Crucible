"""Deterministic, idempotent local DAG execution with retries and run state."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crucible.dedup import DedupConfig, run_dedup
from crucible.dedup.pipeline import write_dedup_report
from crucible.observability import MetricsStore, StageTimer
from crucible.quality import QualityConfig, run_gate, write_report
from crucible.storage import Catalog, Layer
from crucible.utils.hashing import canonical_json, sha256_texts
from crucible.versioning import build_manifest

TaskAction = Callable[["RunContext"], Mapping[str, Any] | None]


class DagError(Exception):
    """Invalid graph or failed task execution."""


@dataclass(frozen=True)
class Task:
    name: str
    action: TaskAction
    dependencies: tuple[str, ...] = ()
    retries: int = 0


@dataclass(frozen=True)
class RunContext:
    root: Path
    run_id: str
    dataset: str


@dataclass
class RunResult:
    run_id: str
    dataset: str
    status: str
    tasks: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "status": self.status,
            "tasks": self.tasks,
        }


def _ordered(tasks: Sequence[Task]) -> list[Task]:
    by_name = {task.name: task for task in tasks}
    if len(by_name) != len(tasks):
        raise DagError("task names must be unique")
    unknown = {dep for task in tasks for dep in task.dependencies if dep not in by_name}
    if unknown:
        raise DagError(f"unknown task dependencies: {sorted(unknown)}")
    result: list[Task] = []
    pending = dict(by_name)
    while pending:
        ready = sorted(
            (
                task
                for task in pending.values()
                if all(d in {t.name for t in result} for d in task.dependencies)
            ),
            key=lambda task: task.name,
        )
        if not ready:
            raise DagError("DAG contains a cycle")
        for task in ready:
            result.append(task)
            del pending[task.name]
    return result


class DagRunner:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.metrics = MetricsStore(self.root)

    def run(self, tasks: Sequence[Task], dataset: str, *, run_id: str | None = None) -> RunResult:
        context = RunContext(self.root, run_id or str(uuid.uuid4()), dataset)
        result = RunResult(context.run_id, dataset, "running")
        run_dir = self.root / "runs" / context.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self._write(run_dir, result)
        try:
            for task in _ordered(tasks):
                self._run_task(task, context, result)
        except Exception:
            result.status = "failed"
            self._write(run_dir, result)
            raise
        result.status = "complete"
        self._write(run_dir, result)
        return result

    def _run_task(self, task: Task, context: RunContext, result: RunResult) -> None:
        error: Exception | None = None
        for attempt in range(1, task.retries + 2):
            try:
                with StageTimer(self.metrics, context.run_id, task.name, attempt=attempt) as timer:
                    payload = dict(task.action(context) or {})
                    timer.input_rows = _optional_int(payload.get("input_rows"))
                    timer.output_rows = _optional_int(payload.get("output_rows"))
                result.tasks.append(
                    {"name": task.name, "status": "complete", "attempts": attempt, **payload}
                )
                return
            except Exception as exc:
                error = exc
        result.tasks.append(
            {
                "name": task.name,
                "status": "failed",
                "attempts": task.retries + 1,
                "error": str(error),
            }
        )
        raise DagError(
            f"task {task.name!r} failed after {task.retries + 1} attempt(s): {error}"
        ) from error

    @staticmethod
    def _write(run_dir: Path, result: RunResult) -> None:
        payload = result.as_dict() | {"updated_at": datetime.now(UTC).isoformat()}
        tmp = run_dir / ".run.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(run_dir / "run.json")


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def pipeline_tasks(quality: QualityConfig, dedup: DedupConfig) -> tuple[Task, ...]:
    """The canonical bronze -> promoted silver -> deduplicated silver DAG."""

    def promote(context: RunContext) -> Mapping[str, Any]:
        catalog = Catalog(context.root)
        result = run_gate(catalog, context.dataset, quality)
        write_report(result, quality, context.root)
        if result.verdict == "blocked":
            raise DagError(f"quality gate blocked {context.dataset!r}")
        return {
            "input_rows": result.input_rows,
            "output_rows": result.promoted_rows,
            "verdict": result.verdict,
        }

    def deduplicate(context: RunContext) -> Mapping[str, Any]:
        result = run_dedup(Catalog(context.root), context.dataset, dedup)
        write_dedup_report(result, context.root)
        return {
            "input_rows": result.input_rows,
            "output_rows": result.kept_rows,
            "removed": len(result.removed_ids),
        }

    return (
        Task("promote", promote, retries=1),
        Task("dedup", deduplicate, ("promote",), retries=1),
    )


def pipeline_fingerprint(
    root: Path, dataset: str, quality: QualityConfig, dedup: DedupConfig
) -> str:
    manifest = build_manifest(Catalog(root), Layer.BRONZE, dataset)
    return sha256_texts(
        [
            manifest.content_hash,
            canonical_json(quality.model_dump()),
            canonical_json(dedup.model_dump()),
        ]
    )


def run_pipeline(
    root: Path, dataset: str, quality: QualityConfig, dedup: DedupConfig, *, force: bool = False
) -> RunResult:
    """Run once per input/config fingerprint unless explicitly forced."""
    fingerprint = pipeline_fingerprint(root, dataset, quality, dedup)
    index_path = Path(root) / "runs" / "pipeline-index.json"
    index: dict[str, str] = json.loads(index_path.read_text()) if index_path.exists() else {}
    existing = index.get(fingerprint)
    if existing and not force:
        record = json.loads((Path(root) / "runs" / existing / "run.json").read_text())
        return RunResult(record["run_id"], record["dataset"], "skipped", record["tasks"])
    result = DagRunner(root).run(pipeline_tasks(quality, dedup), dataset)
    index[fingerprint] = result.run_id
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    tmp.replace(index_path)
    return result


def list_runs(root: Path, limit: int = 100) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted((Path(root) / "runs").glob("*/run.json")):
        try:
            runs.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return runs[-limit:]
