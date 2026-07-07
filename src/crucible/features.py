"""Minimal offline feature store with point-in-time-correct joins.

A *feature view* is a materialized table of ``(entity, event_timestamp,
feature columns)`` rows under ``<root>/features/<name>/``. The one join
this store offers is the only one that is safe for training data:

    for each spine row (entity, t), attach the latest feature row of that
    entity with feature timestamp <= t

implemented as a DuckDB ASOF LEFT JOIN. Joining "the latest value" without
the as-of bound — what a naive SQL join does — silently leaks the future
into training rows; the tests demonstrate exactly that failure mode, and
``assert_no_leakage`` re-checks the invariant on any joined frame as a
guard (belt and suspenders: the ASOF join is correct by construction, the
guard catches misuse and future regressions).

Offline/online parity: ``get_latest(view, entity)`` answers "what would
the online store serve right now" and must equal the PIT join evaluated at
t = infinity, which the parity test asserts.

Timestamps are ISO-8601 UTC strings end to end (they sort correctly and
DuckDB casts them); entities are strings.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from crucible.storage import Catalog, table_content_hash


class LeakageError(Exception):
    """A joined feature row postdates its spine row's timestamp."""


@dataclass(frozen=True, slots=True)
class FeatureView:
    name: str
    entity_column: str
    timestamp_column: str
    feature_columns: list[str]
    n_rows: int
    content_hash: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeatureStore:
    """Feature views materialized as Parquet under the catalog root."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.root = catalog.root / "features"

    # -- registry -------------------------------------------------------------

    def _view_dir(self, name: str) -> Path:
        return self.root / name

    def register(
        self,
        name: str,
        table: pa.Table,
        entity_column: str,
        timestamp_column: str,
    ) -> FeatureView:
        """Materialize a feature view; rows must carry entity + timestamp."""
        for column in (entity_column, timestamp_column):
            if column not in table.column_names:
                raise ValueError(f"feature view {name!r} is missing column {column!r}")
        feature_columns = [
            column
            for column in table.column_names
            if column not in (entity_column, timestamp_column)
        ]
        if not feature_columns:
            raise ValueError(f"feature view {name!r} has no feature columns")

        view = FeatureView(
            name=name,
            entity_column=entity_column,
            timestamp_column=timestamp_column,
            feature_columns=feature_columns,
            n_rows=table.num_rows,
            content_hash=table_content_hash(table),
        )
        directory = self._view_dir(name)
        directory.mkdir(parents=True, exist_ok=True)
        import pyarrow.parquet as pq

        pq.write_table(table, directory / "view.parquet")
        (directory / "view.json").write_text(json.dumps(view.as_dict(), indent=2) + "\n")
        return view

    def load_view(self, name: str) -> tuple[FeatureView, pa.Table]:
        directory = self._view_dir(name)
        meta_path = directory / "view.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no feature view named {name!r} under {self.root}")
        raw = json.loads(meta_path.read_text())
        import pyarrow.parquet as pq

        return FeatureView(**raw), pq.read_table(directory / "view.parquet")

    def views(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(child.name for child in self.root.iterdir() if child.is_dir())

    # -- point-in-time join ---------------------------------------------------

    def point_in_time_join(
        self,
        spine: pa.Table,
        view_name: str,
        spine_entity_column: str,
        spine_timestamp_column: str,
    ) -> pa.Table:
        """Attach, per spine row, the latest feature row with ts <= spine ts."""
        view, feature_table = self.load_view(view_name)
        import duckdb

        feature_selects = ", ".join(
            f'f."{column}" AS "{view_name}__{column}"' for column in view.feature_columns
        )
        with duckdb.connect() as conn:
            conn.register("spine", spine)
            conn.register("features", feature_table)
            joined = conn.execute(f"""
                SELECT s.*,
                       {feature_selects},
                       f."{view.timestamp_column}" AS "{view_name}__feature_ts"
                FROM spine s
                ASOF LEFT JOIN features f
                  ON s."{spine_entity_column}" = f."{view.entity_column}"
                 AND f."{view.timestamp_column}" <= s."{spine_timestamp_column}"
                """).arrow()
        result: pa.Table = pa.table(joined)
        assert_no_leakage(result, spine_timestamp_column, f"{view_name}__feature_ts")
        return result

    def get_latest(self, view_name: str, entity: str) -> dict[str, Any] | None:
        """What the online store would serve for this entity right now."""
        view, feature_table = self.load_view(view_name)
        latest: dict[str, Any] | None = None
        for row in feature_table.to_pylist():
            if str(row[view.entity_column]) != entity:
                continue
            if latest is None or str(row[view.timestamp_column]) > str(
                latest[view.timestamp_column]
            ):
                latest = row
        return latest


def assert_no_leakage(
    joined: pa.Table, spine_timestamp_column: str, feature_timestamp_column: str
) -> None:
    """Every attached feature must be from the spine row's past (or absent)."""
    spine_ts = joined.column(spine_timestamp_column).to_pylist()
    feature_ts = joined.column(feature_timestamp_column).to_pylist()
    for index, (spine_value, feature_value) in enumerate(zip(spine_ts, feature_ts, strict=True)):
        if feature_value is not None and str(feature_value) > str(spine_value):
            raise LeakageError(
                f"row {index}: feature timestamp {feature_value} postdates "
                f"spine timestamp {spine_value}"
            )


def source_rollup_features(
    table: pa.Table,
    entity_column: str = "source",
    timestamp_column: str = "timestamp",
    text_column: str = "text",
) -> pa.Table:
    """Cumulative per-source stats known *as of* each record's timestamp.

    For every (source, t) event: how many docs this source has produced so
    far and their running mean word count — the kind of slowly-changing
    aggregate a curriculum or mixture policy would consume. Cumulative by
    construction, so a PIT join over these features can never peek forward.
    """
    events: list[tuple[str, str, int]] = sorted(
        (
            str(row[timestamp_column]),
            str(row[entity_column]),
            len(str(row[text_column] or "").split()),
        )
        for row in table.to_pylist()
    )
    counts: dict[str, int] = {}
    word_totals: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for ts, entity, words in events:
        counts[entity] = counts.get(entity, 0) + 1
        word_totals[entity] = word_totals.get(entity, 0) + words
        out.append(
            {
                "source": entity,
                "timestamp": ts,
                "docs_so_far": counts[entity],
                "mean_words_so_far": round(word_totals[entity] / counts[entity], 3),
            }
        )
    return pa.Table.from_pylist(out)
