"""Content-addressed dataset manifests and version snapshots.

A dataset's identity is its manifest content hash: parts are already
content-addressed by name, so the manifest hash is a hash over the sorted
``part-name:part-file-sha256`` lines — stable across read order, machines,
and time. A *snapshot* pins one pipeline stage run:

    snapshot_id = hash(stage, config hash, input manifest hashes,
                       code version, output manifest hash)

which is exactly the reproducibility contract: same config + same inputs +
same code must rebuild the same output, and ``verify_snapshot`` checks the
bytes on disk still match. Snapshots are written automatically by the gate
and dedup stages (the config-driven stages); ingestion emits lineage events
but no snapshot, since its "config" is an external source.

Snapshot JSON lives under ``<root>/versions/<dataset>/`` — operational
metadata, like the ingest log: excluded from dataset identity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel

from crucible import __version__
from crucible.storage import Catalog, Layer
from crucible.utils.hashing import canonical_json, sha256_file, sha256_text, sha256_texts


@dataclass(frozen=True, slots=True)
class PartInfo:
    name: str
    sha256: str
    rows: int


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    layer: str
    dataset: str
    n_rows: int
    schema: list[list[str]]  # [name, type] pairs, JSON-friendly
    parts: list[PartInfo]
    content_hash: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_manifest(catalog: Catalog, layer: Layer, dataset: str) -> DatasetManifest:
    part_paths = catalog.parts(layer, dataset)
    if not part_paths:
        raise ValueError(f"no parts to manifest in {layer.value}/{dataset}")
    parts = [
        PartInfo(
            name=path.name,
            sha256=sha256_file(path),
            rows=pq.ParquetFile(path).metadata.num_rows,
        )
        for path in part_paths
    ]
    schema = [[field.name, str(field.type)] for field in pq.read_schema(part_paths[0])]
    content_hash = sha256_texts(
        f"{part.name}:{part.sha256}" for part in sorted(parts, key=lambda p: p.name)
    )
    return DatasetManifest(
        layer=layer.value,
        dataset=dataset,
        n_rows=sum(part.rows for part in parts),
        schema=schema,
        parts=parts,
        content_hash=content_hash,
    )


def snapshot_stage(
    catalog: Catalog,
    stage: str,
    config: BaseModel | dict[str, Any],
    input_manifests: list[DatasetManifest],
    output: tuple[Layer, str],
) -> dict[str, Any]:
    """Record one stage run as a version snapshot; returns the record.

    Input manifests are passed in (not rebuilt here) because for in-place
    stages like dedup the input no longer exists on disk by the time the
    stage finishes — the caller manifests it before rewriting.
    """
    config_dict = config.model_dump(mode="json") if isinstance(config, BaseModel) else config
    config_hash = sha256_text(canonical_json(config_dict))
    output_manifest = build_manifest(catalog, *output)
    snapshot_id = sha256_text(
        canonical_json(
            {
                "stage": stage,
                "config_hash": config_hash,
                "inputs": [manifest.content_hash for manifest in input_manifests],
                "code_version": __version__,
                "output": output_manifest.content_hash,
            }
        )
    )[:12]
    record: dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "stage": stage,
        "dataset": output[1],
        "created_at": datetime.now(UTC).isoformat(),
        "code_version": __version__,
        "config": config_dict,
        "config_hash": config_hash,
        "inputs": [
            {
                "layer": manifest.layer,
                "dataset": manifest.dataset,
                "content_hash": manifest.content_hash,
                "n_rows": manifest.n_rows,
            }
            for manifest in input_manifests
        ],
        "output": output_manifest.as_dict(),
    }
    directory = catalog.root / "versions" / output[1]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stage}-{snapshot_id}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def list_snapshots(root: Path, dataset: str) -> list[dict[str, Any]]:
    """All snapshots for a dataset, newest last (by created_at)."""
    directory = root / "versions" / dataset
    if not directory.is_dir():
        return []
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    return sorted(records, key=lambda record: str(record["created_at"]))


def verify_snapshot(catalog: Catalog, record: dict[str, Any]) -> tuple[bool, str]:
    """Do the bytes on disk still match what the snapshot pinned?"""
    output = record["output"]
    layer, dataset = Layer(output["layer"]), str(output["dataset"])
    try:
        current = build_manifest(catalog, layer, dataset)
    except ValueError as exc:
        return False, str(exc)
    if current.content_hash != output["content_hash"]:
        return False, (
            f"{layer.value}/{dataset} content hash {current.content_hash[:12]}... "
            f"!= snapshot {str(output['content_hash'])[:12]}..."
        )
    return True, "ok"
