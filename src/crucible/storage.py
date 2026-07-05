"""Medallion storage: layered Parquet datasets with a DuckDB catalog.

Layers are contracts (see docs/data_contracts.md): bronze is raw and
immutable, silver is validated and deduplicated, gold is curated mixtures,
quarantine holds records rejected by quality gates. A dataset in a layer is
a directory of immutable Parquet part files; parts are content-addressed by
the writer (see ``crucible.ingest.land``) and written atomically (tmp file +
rename) so a crashed writer can never leave a torn part visible.

The catalog root is a local directory today. The layout and all access
going through :class:`Catalog` are deliberately object-store-shaped
(flat immutable parts, no renames after publish, no appends), so pointing
the same code at S3/MinIO later is a path-prefix change plus DuckDB's
httpfs extension, not a redesign.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

# Dataset names become DuckDB view names (``<layer>_<dataset>``), so keep
# them SQL-identifier-safe by construction.
_DATASET_NAME = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


class StorageError(Exception):
    """Invalid catalog operation (bad name, missing dataset, torn write)."""


class Layer(StrEnum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    QUARANTINE = "quarantine"


def _validate_dataset_name(dataset: str) -> str:
    if not _DATASET_NAME.match(dataset):
        raise StorageError(
            f"invalid dataset name {dataset!r}: must match {_DATASET_NAME.pattern}"
        )
    return dataset


class Catalog:
    """Filesystem-backed medallion catalog with SQL views over every dataset."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- layout -------------------------------------------------------------

    def dataset_dir(self, layer: Layer, dataset: str) -> Path:
        return self.root / layer.value / _validate_dataset_name(dataset)

    def parts(self, layer: Layer, dataset: str) -> list[Path]:
        """Published Parquet parts of a dataset, in stable (sorted) order."""
        directory = self.dataset_dir(layer, dataset)
        return sorted(directory.glob("*.parquet"))

    def datasets(self, layer: Layer) -> list[str]:
        layer_dir = self.root / layer.value
        if not layer_dir.is_dir():
            return []
        return sorted(child.name for child in layer_dir.iterdir() if child.is_dir())

    # -- writes -------------------------------------------------------------

    def write_part(self, table: pa.Table, layer: Layer, dataset: str, part_name: str) -> Path:
        """Atomically publish one immutable Parquet part.

        The part is written to a ``.tmp`` sibling and renamed into place, so
        readers (and DuckDB globs) can never observe a half-written part.
        Re-publishing the same part name is allowed and atomic — with
        content-addressed names, identical name implies identical bytes.
        """
        directory = self.dataset_dir(layer, dataset)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{part_name}.parquet"
        tmp = directory / f".{part_name}.parquet.tmp"
        pq.write_table(table, tmp)
        tmp.rename(target)
        return target

    # -- reads --------------------------------------------------------------

    def read(self, layer: Layer, dataset: str) -> pa.Table:
        parts = self.parts(layer, dataset)
        if not parts:
            raise StorageError(f"no parts in {layer.value}/{dataset}")
        return pa.concat_tables(pq.read_table(part) for part in parts)

    def row_count(self, layer: Layer, dataset: str) -> int:
        return sum(
            pq.ParquetFile(part).metadata.num_rows for part in self.parts(layer, dataset)
        )

    # -- SQL ----------------------------------------------------------------

    def connect(self) -> duckdb.DuckDBPyConnection:
        """In-memory DuckDB connection with a view per dataset.

        View naming: ``<layer>_<dataset>``, e.g. ``bronze_synth``.
        """
        conn = duckdb.connect()
        for layer in Layer:
            for dataset in self.datasets(layer):
                if not self.parts(layer, dataset):
                    continue
                glob = str(self.dataset_dir(layer, dataset) / "*.parquet")
                escaped = glob.replace("'", "''")
                conn.execute(
                    f'CREATE VIEW "{layer.value}_{dataset}" AS '
                    f"SELECT * FROM read_parquet('{escaped}')"
                )
        return conn

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Run SQL over the catalog views; rows come back as dicts."""
        with self.connect() as conn:
            cursor = conn.execute(sql)
            columns = [description[0] for description in cursor.description or []]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def summary(self) -> dict[str, dict[str, dict[str, int]]]:
        """Per-layer, per-dataset part and row counts (for `crucible catalog`)."""
        out: dict[str, dict[str, dict[str, int]]] = {}
        for layer in Layer:
            layer_summary = {
                dataset: {
                    "parts": len(self.parts(layer, dataset)),
                    "rows": self.row_count(layer, dataset),
                }
                for dataset in self.datasets(layer)
            }
            if layer_summary:
                out[layer.value] = layer_summary
        return out
