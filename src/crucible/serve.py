"""Read-only FastAPI metadata service over a Crucible catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crucible.lineage import LineageGraph
from crucible.observability import MetricsStore
from crucible.orchestrate import list_runs
from crucible.storage import Catalog
from crucible.versioning import list_snapshots


def _read_report(root: Path, kind: str, dataset: str) -> dict[str, Any]:
    path = root / "reports" / kind / f"{dataset}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report is not an object: {path}")
    return value


def create_app(root: Path) -> Any:
    """Application factory. FastAPI stays optional until serving is requested."""
    try:
        from fastapi import FastAPI, HTTPException, Query
    except ImportError as exc:  # pragma: no cover - exercised in torch-free style envs
        raise RuntimeError("serving requires `pip install crucible-data[serve]`") from exc

    catalog_root = Path(root)
    app = FastAPI(
        title="Crucible Metadata API",
        version="0.8.0",
        description="Read-only catalog, quality, lineage, run, and metric metadata.",
    )

    @app.get("/healthz", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/catalog", tags=["catalog"])
    def catalog_summary() -> dict[str, Any]:
        return Catalog(catalog_root).summary()

    @app.get("/v1/versions/{dataset}", tags=["catalog"])
    def versions(dataset: str) -> list[dict[str, Any]]:
        return list_snapshots(catalog_root, dataset)

    @app.get("/v1/lineage", tags=["lineage"])
    def lineage() -> dict[str, Any]:
        graph = LineageGraph.from_root(catalog_root)
        return {
            "datasets": sorted(graph.datasets),
            "jobs": sorted(graph.jobs),
            "edges": graph.edges(),
        }

    @app.get("/v1/reports/{kind}/{dataset}", tags=["reports"])
    def report(kind: str, dataset: str) -> dict[str, Any]:
        if kind not in {"quality", "dedup"}:
            raise HTTPException(status_code=400, detail="kind must be quality or dedup")
        try:
            return _read_report(catalog_root, kind, dataset)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="report not found") from exc

    @app.get("/v1/runs", tags=["operations"])
    def runs(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
        return list_runs(catalog_root, limit)

    @app.get("/v1/metrics", tags=["operations"])
    def metrics(
        run_id: str | None = None, limit: int = Query(default=100, ge=1, le=1000)
    ) -> list[dict[str, Any]]:
        return MetricsStore(catalog_root).list(run_id=run_id, limit=limit)

    return app
