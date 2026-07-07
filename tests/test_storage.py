from pathlib import Path

import pyarrow as pa
import pytest

from crucible.storage import Catalog, Layer, StorageError


def _table(ids: list[str]) -> pa.Table:
    return pa.table({"id": ids, "text": [f"doc {i}" for i in ids]})


def test_write_part_and_read_back(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path)
    catalog.write_part(_table(["a", "b"]), Layer.BRONZE, "demo", "part-0001")
    catalog.write_part(_table(["c"]), Layer.BRONZE, "demo", "part-0002")
    table = catalog.read(Layer.BRONZE, "demo")
    assert table.num_rows == 3
    assert catalog.row_count(Layer.BRONZE, "demo") == 3
    assert [p.name for p in catalog.parts(Layer.BRONZE, "demo")] == [
        "part-0001.parquet",
        "part-0002.parquet",
    ]


def test_write_is_atomic_no_tmp_visible(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path)
    catalog.write_part(_table(["a"]), Layer.BRONZE, "demo", "part-0001")
    files = {p.name for p in catalog.dataset_dir(Layer.BRONZE, "demo").iterdir()}
    assert files == {"part-0001.parquet"}  # no .tmp residue


def test_republish_same_part_is_idempotent(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path)
    catalog.write_part(_table(["a"]), Layer.BRONZE, "demo", "part-0001")
    catalog.write_part(_table(["a"]), Layer.BRONZE, "demo", "part-0001")
    assert catalog.row_count(Layer.BRONZE, "demo") == 1


def test_dataset_name_validation(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path)
    for bad in ["Bad", "1abc", "a b", "a;drop", "a" * 64]:
        with pytest.raises(StorageError, match="invalid dataset name"):
            catalog.dataset_dir(Layer.BRONZE, bad)


def test_read_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="no parts"):
        Catalog(tmp_path).read(Layer.BRONZE, "nope")


def test_datasets_listing(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path)
    assert catalog.datasets(Layer.BRONZE) == []
    catalog.write_part(_table(["a"]), Layer.BRONZE, "beta", "part-0001")
    catalog.write_part(_table(["a"]), Layer.BRONZE, "alpha", "part-0001")
    catalog.write_part(_table(["a"]), Layer.SILVER, "alpha", "part-0001")
    assert catalog.datasets(Layer.BRONZE) == ["alpha", "beta"]
    assert catalog.datasets(Layer.SILVER) == ["alpha"]


def test_query_over_views(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path)
    catalog.write_part(_table(["a", "b"]), Layer.BRONZE, "demo", "part-0001")
    catalog.write_part(_table(["c"]), Layer.SILVER, "demo", "part-0001")
    rows = catalog.query(
        "SELECT (SELECT count(*) FROM bronze_demo) AS b, (SELECT count(*) FROM silver_demo) AS s"
    )
    assert rows == [{"b": 2, "s": 1}]


def test_summary(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path)
    catalog.write_part(_table(["a", "b"]), Layer.BRONZE, "demo", "part-0001")
    assert catalog.summary() == {"bronze": {"demo": {"parts": 1, "rows": 2}}}
