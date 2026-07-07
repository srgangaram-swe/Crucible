import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from crucible.ingest import IngestError, JsonlSource, land
from crucible.storage import Catalog, Layer
from crucible.synth import SynthConfig, generate_corpus, write_jsonl


class ListSource:
    """Test source over in-memory tables, optionally failing mid-stream."""

    def __init__(self, tables: list[pa.Table], fail_after: int | None = None) -> None:
        self.tables = tables
        self.fail_after = fail_after

    def batches(self) -> Iterator[pa.Table]:
        for i, table in enumerate(self.tables):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("simulated source crash")
            yield table


@pytest.fixture
def corpus_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    write_jsonl(generate_corpus(SynthConfig(seed=5, n_docs=150)), path)
    return path


def test_land_and_query(tmp_path: Path, corpus_jsonl: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    result = land(JsonlSource(corpus_jsonl, batch_size=40), catalog, "synth", "test")
    assert result.rows_written == 150
    assert result.parts_written == 4  # 40+40+40+30
    rows = catalog.query("SELECT count(*) AS n, count(DISTINCT id) AS ids FROM bronze_synth")
    assert rows == [{"n": 150, "ids": 150}]


def test_land_is_idempotent(tmp_path: Path, corpus_jsonl: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    source = JsonlSource(corpus_jsonl, batch_size=40)
    land(source, catalog, "synth", "test")
    again = land(source, catalog, "synth", "test")
    assert again.parts_written == 0
    assert again.rows_written == 0
    assert again.parts_skipped == 4
    assert again.rows_skipped == 150
    assert catalog.row_count(Layer.BRONZE, "synth") == 150


def test_land_resumes_after_crash(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    tables = [pa.table({"id": [f"r{i}-{j}" for j in range(5)]}) for i in range(4)]
    with pytest.raises(RuntimeError, match="simulated source crash"):
        land(ListSource(tables, fail_after=2), catalog, "demo", "test")
    assert catalog.row_count(Layer.BRONZE, "demo") == 10  # 2 batches landed

    result = land(ListSource(tables), catalog, "demo", "test")
    assert result.parts_skipped == 2  # the two already landed
    assert result.parts_written == 2
    assert catalog.row_count(Layer.BRONZE, "demo") == 20


def test_land_rejects_schema_drift(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    land(ListSource([pa.table({"id": ["a"], "text": ["t"]})]), catalog, "demo", "test")
    with pytest.raises(IngestError, match="schema drift"):
        land(ListSource([pa.table({"id": ["b"], "other": ["x"]})]), catalog, "demo", "test")


def test_land_forgives_column_order(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    land(ListSource([pa.table({"id": ["a"], "text": ["t"]})]), catalog, "demo", "test")
    result = land(ListSource([pa.table({"text": ["u"], "id": ["b"]})]), catalog, "demo", "test")
    assert result.rows_written == 1
    table = catalog.read(Layer.BRONZE, "demo")
    assert table.column_names == ["id", "text"]
    assert table.num_rows == 2


def test_land_promotes_all_null_columns_to_string(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    first = pa.Table.from_pylist([{"id": "a", "ref": None}])  # ref inferred as null type
    land(ListSource([first]), catalog, "demo", "test")
    schema = pq.read_schema(catalog.parts(Layer.BRONZE, "demo")[0])
    assert schema.field("ref").type == pa.string()
    # A later batch with real string values conforms.
    result = land(
        ListSource([pa.Table.from_pylist([{"id": "b", "ref": "a"}])]), catalog, "demo", "test"
    )
    assert result.rows_written == 1


def test_land_skips_empty_batches(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    empty = pa.table({"id": pa.array([], type=pa.string())})
    result = land(ListSource([empty, pa.table({"id": ["a"]})]), catalog, "demo", "test")
    assert result.parts_written == 1
    assert result.rows_written == 1


def test_ingest_log_records_parts(tmp_path: Path, corpus_jsonl: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    land(JsonlSource(corpus_jsonl, batch_size=75), catalog, "synth", "smoke-source")
    log_path = catalog.dataset_dir(Layer.BRONZE, "synth") / "_ingest_log.jsonl"
    entries = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(entries) == 2
    assert all(entry["source"] == "smoke-source" for entry in entries)
    assert all(entry["n_rows"] == 75 for entry in entries)
    parts_on_disk = {p.name for p in catalog.parts(Layer.BRONZE, "synth")}
    assert {entry["part"] for entry in entries} == parts_on_disk
