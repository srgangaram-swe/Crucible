"""Idempotent landing of source batches into bronze.

Exactly-once-ish, by construction rather than coordination:

1. every batch is content-addressed (SHA-256 of its canonical rows),
2. the part file is named by that hash and published atomically
   (tmp + rename, see :meth:`crucible.storage.Catalog.write_part`),
3. the ingest log records every published part, and already-landed hashes
   are skipped.

Under at-least-once delivery — retries, replays, crashed runs — re-landing
the same batch is a no-op, so delivery duplicates collapse to exactly-once
*landing*. Row-level duplicates across batches are a data property, not a
delivery property; removing them is the dedup stage's job (Phase 3).

The ingest log (``_ingest_log.jsonl``) is operational metadata: it records
what landed and when. Dataset *identity* is the content of the Parquet
parts alone; the log (with its wall-clock timestamps) is excluded from
content hashes on purpose.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from crucible import __version__
from crucible.ingest.sources import Source
from crucible.lineage import dataset_ref, emit_event
from crucible.storage import Catalog, Layer, table_content_hash
from crucible.versioning import build_manifest

_LOG_NAME = "_ingest_log.jsonl"


class IngestConfig(BaseModel):
    """One ingest run: land ``input`` into ``bronze/<dataset>`` under ``root``."""

    input: Path
    dataset: str
    root: Path = Path("data/crucible")
    fmt: str = "auto"
    batch_size: int = Field(default=1000, ge=1)
    source_name: str | None = None
    via_stream: bool = False  # replay through the in-memory broker (jsonl only)


class IngestError(Exception):
    """A batch could not be landed (schema drift, uncastable types)."""


@dataclass(frozen=True, slots=True)
class IngestResult:
    dataset: str
    layer: str
    parts_written: int
    parts_skipped: int
    rows_written: int
    rows_skipped: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def batch_hash(table: pa.Table) -> str:
    """Order-sensitive content hash of a batch's canonical rows."""
    return table_content_hash(table)


def _log_path(catalog: Catalog, dataset: str) -> Path:
    return catalog.dataset_dir(Layer.BRONZE, dataset) / _LOG_NAME


def _landed_hashes(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    hashes = set()
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                hashes.add(str(json.loads(line)["batch_hash"]))
    return hashes


def _canonical_schema(catalog: Catalog, dataset: str) -> pa.Schema | None:
    """The dataset's schema is the schema of its first published part."""
    parts = catalog.parts(Layer.BRONZE, dataset)
    return pq.read_schema(parts[0]) if parts else None


def _promote_nulls(schema: pa.Schema) -> pa.Schema:
    """All-null inferred columns (e.g. a nullable field that happens to be
    empty in the first batch) default to string rather than Arrow's null
    type, so later non-null batches still conform."""
    return pa.schema(
        [
            pa.field(field.name, pa.string()) if pa.types.is_null(field.type) else field
            for field in schema
        ]
    )


def _normalize(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Cast a batch to the dataset schema; column order is forgiven, column
    set and castability are not."""
    if table.schema == schema:
        return table
    if set(table.column_names) != set(schema.names):
        raise IngestError(
            f"schema drift: batch columns {sorted(table.column_names)} != "
            f"dataset columns {sorted(schema.names)}"
        )
    try:
        return table.select(schema.names).cast(schema)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise IngestError(f"batch is not castable to dataset schema: {exc}") from exc


def land(
    source: Source,
    catalog: Catalog,
    dataset: str,
    source_name: str = "unknown",
) -> IngestResult:
    """Land all batches from ``source`` into ``bronze/<dataset>``."""
    catalog.dataset_dir(Layer.BRONZE, dataset).mkdir(parents=True, exist_ok=True)
    log_path = _log_path(catalog, dataset)
    seen = _landed_hashes(log_path)
    schema = _canonical_schema(catalog, dataset)

    parts_written = parts_skipped = rows_written = rows_skipped = 0
    with log_path.open("a", encoding="utf-8") as log:
        for raw in source.batches():
            if raw.num_rows == 0:
                continue
            if schema is None:
                schema = _promote_nulls(raw.schema)
            table = _normalize(raw, schema)
            digest = batch_hash(table)
            if digest in seen:
                parts_skipped += 1
                rows_skipped += table.num_rows
                continue
            part = catalog.write_part(table, Layer.BRONZE, dataset, f"part-{digest[:16]}")
            entry = {
                "batch_hash": digest,
                "part": part.name,
                "n_rows": table.num_rows,
                "source": source_name,
                "crucible_version": __version__,
                "ingested_at": datetime.now(UTC).isoformat(),
            }
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.flush()
            seen.add(digest)
            parts_written += 1
            rows_written += table.num_rows

    result = IngestResult(
        dataset=dataset,
        layer=Layer.BRONZE.value,
        parts_written=parts_written,
        parts_skipped=parts_skipped,
        rows_written=rows_written,
        rows_skipped=rows_skipped,
    )
    if catalog.parts(Layer.BRONZE, dataset):
        manifest = build_manifest(catalog, Layer.BRONZE, dataset)
        emit_event(
            catalog.root,
            job=f"ingest:{dataset}",
            inputs=[{"namespace": "external", "name": source_name, "facets": {}}],
            outputs=[dataset_ref(f"bronze/{dataset}", manifest.content_hash, manifest.n_rows)],
            facets={"rows_written": rows_written, "parts_skipped": parts_skipped},
        )
    return result
