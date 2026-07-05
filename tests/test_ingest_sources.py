import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from crucible.ingest import CsvSource, JsonlSource, ParquetSource, open_source

ROWS = [{"id": f"r{i}", "value": i} for i in range(10)]


def _collect(source: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for table in source.batches():  # type: ignore[attr-defined]
        rows.extend(table.to_pylist())
    return rows


def test_jsonl_source_batches(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in ROWS))
    source = JsonlSource(path, batch_size=3)
    tables = list(source.batches())
    assert [t.num_rows for t in tables] == [3, 3, 3, 1]
    assert _collect(source) == ROWS


def test_jsonl_source_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "a"}\n\n\n{"id": "b"}\n')
    assert _collect(JsonlSource(path)) == [{"id": "a"}, {"id": "b"}]


def test_parquet_source_batches(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist(ROWS), path)
    source = ParquetSource(path, batch_size=4)
    tables = list(source.batches())
    assert [t.num_rows for t in tables] == [4, 4, 2]
    assert _collect(source) == ROWS


def test_csv_source_batches(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("id,value\n" + "".join(f"r{i},{i}\n" for i in range(10)))
    source = CsvSource(path, batch_size=6)
    tables = list(source.batches())
    assert [t.num_rows for t in tables] == [6, 4]
    assert _collect(source) == ROWS


def test_open_source_by_suffix_and_fmt(tmp_path: Path) -> None:
    jsonl = tmp_path / "a.jsonl"
    jsonl.write_text('{"id": "a"}\n')
    assert isinstance(open_source(jsonl), JsonlSource)
    assert isinstance(open_source(jsonl, fmt="csv"), CsvSource)
    with pytest.raises(ValueError, match="unknown format"):
        open_source(jsonl, fmt="xml")
    with pytest.raises(ValueError, match="cannot infer format"):
        open_source(tmp_path / "a.tdms")


def test_hf_source_with_injected_dataset() -> None:
    datasets = pytest.importorskip("datasets")
    from crucible.ingest import HFSource

    dataset = datasets.Dataset.from_dict({"id": ["a", "b", "c"], "value": [1, 2, 3]})
    tables = list(HFSource(dataset, batch_size=2).batches())
    assert [t.num_rows for t in tables] == [2, 1]
    assert tables[0].column_names == ["id", "value"]
