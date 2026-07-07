"""Lineage: OpenLineage-inspired run events and a queryable local graph.

Every pipeline stage appends a COMPLETE event to
``<root>/lineage/events.jsonl`` naming its input and output datasets with
their manifest content hashes. The event shape follows OpenLineage's
run-event vocabulary (eventType/run/job/inputs/outputs with facets) so the
emitter could be pointed at a real OpenLineage backend (Marquez) later,
but this is deliberately a local, append-only file — see limitations.md.

``LineageGraph`` folds the log into datasets + jobs with edges, keeping the
latest event per job name (stages are idempotent, so re-runs supersede).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_EVENTS_FILE = "events.jsonl"


def dataset_ref(
    name: str, content_hash: str | None = None, rows: int | None = None
) -> dict[str, Any]:
    """An event input/output entry, e.g. name='bronze/synth'."""
    facets: dict[str, Any] = {}
    if content_hash is not None:
        facets["contentHash"] = content_hash
    if rows is not None:
        facets["rowCount"] = rows
    return {"namespace": "crucible", "name": name, "facets": facets}


def emit_event(
    root: Path,
    job: str,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    facets: dict[str, Any] | None = None,
) -> None:
    directory = root / "lineage"
    directory.mkdir(parents=True, exist_ok=True)
    event = {
        "eventType": "COMPLETE",
        "eventTime": datetime.now(UTC).isoformat(),
        "run": {"runId": str(uuid.uuid4()), "facets": facets or {}},
        "job": {"namespace": "crucible", "name": job},
        "inputs": inputs,
        "outputs": outputs,
    }
    with (directory / _EVENTS_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


@dataclass
class LineageGraph:
    """Datasets and jobs with directed edges, from the event log."""

    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)  # latest event per job
    datasets: set[str] = field(default_factory=set)

    @classmethod
    def from_root(cls, root: Path) -> LineageGraph:
        graph = cls()
        events_path = root / "lineage" / _EVENTS_FILE
        if not events_path.exists():
            return graph
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                graph.jobs[str(event["job"]["name"])] = event
                for entry in [*event["inputs"], *event["outputs"]]:
                    graph.datasets.add(str(entry["name"]))
        return graph

    def edges(self) -> list[tuple[str, str]]:
        """(from, to) pairs; jobs are rendered as `job:<name>` nodes."""
        out: list[tuple[str, str]] = []
        for name, event in sorted(self.jobs.items()):
            job_node = f"job:{name}"
            out.extend((str(entry["name"]), job_node) for entry in event["inputs"])
            out.extend((job_node, str(entry["name"])) for entry in event["outputs"])
        return out

    def upstream(self, dataset: str) -> set[str]:
        """All datasets that (transitively) feed ``dataset``."""
        parents: dict[str, set[str]] = {}
        for source, target in self.edges():
            parents.setdefault(target, set()).add(source)
        seen: set[str] = set()
        frontier = [dataset]
        while frontier:
            node = frontier.pop()
            for parent in parents.get(node, set()):
                key = parent.removeprefix("job:")
                if parent.startswith("job:"):
                    frontier.append(parent)  # pass through job nodes
                elif key not in seen:
                    seen.add(key)
                    frontier.append(parent)
        return seen

    def to_mermaid(self) -> str:
        lines = ["flowchart LR"]
        for dataset in sorted(self.datasets):
            node_id = dataset.replace("/", "_")
            lines.append(f'    {node_id}["{dataset}"]')
        for name in sorted(self.jobs):
            node_id = f"job_{name}".replace("/", "_").replace(":", "_").replace("-", "_")
            lines.append(f'    {node_id}(["{name}"])')
        for source, target in self.edges():

            def mermaid_id(node: str) -> str:
                if node.startswith("job:"):
                    return "job_" + node.removeprefix("job:").replace("/", "_").replace(
                        ":", "_"
                    ).replace("-", "_")
                return node.replace("/", "_")

            lines.append(f"    {mermaid_id(source)} --> {mermaid_id(target)}")
        return "\n".join(lines)
