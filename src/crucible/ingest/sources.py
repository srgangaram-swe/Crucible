"""Batch source connectors.

A source yields Arrow micro-batches; the lander neither knows nor cares
where they came from, which is what lets the streaming path (see
:mod:`crucible.ingest.stream`) reuse the exact same landing code.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

if TYPE_CHECKING:
    import datasets as hf_datasets


class Source(Protocol):
    """Anything that yields Arrow tables in deterministic micro-batches."""

    def batches(self) -> Iterator[pa.Table]: ...


def _slice_table(table: pa.Table, batch_size: int) -> Iterator[pa.Table]:
    for start in range(0, table.num_rows, batch_size):
        yield table.slice(start, batch_size)


class ParquetSource:
    def __init__(self, path: Path, batch_size: int = 1000) -> None:
        self.path = path
        self.batch_size = batch_size

    def batches(self) -> Iterator[pa.Table]:
        parquet_file = pq.ParquetFile(self.path)
        for batch in parquet_file.iter_batches(batch_size=self.batch_size):
            yield pa.Table.from_batches([batch])


class CsvSource:
    def __init__(self, path: Path, batch_size: int = 1000) -> None:
        self.path = path
        self.batch_size = batch_size

    def batches(self) -> Iterator[pa.Table]:
        yield from _slice_table(pacsv.read_csv(self.path), self.batch_size)


class JsonlSource:
    """Newline-delimited JSON; the native format of the synthetic corpus."""

    def __init__(self, path: Path, batch_size: int = 1000) -> None:
        self.path = path
        self.batch_size = batch_size

    def batches(self) -> Iterator[pa.Table]:
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if len(rows) >= self.batch_size:
                    yield pa.Table.from_pylist(rows)
                    rows = []
        if rows:
            yield pa.Table.from_pylist(rows)


class HFSource:
    """Hugging Face `datasets` connector (requires the ``hf`` extra).

    Accepts a pre-loaded dataset object so tests and callers can inject
    local datasets without touching the network; :meth:`from_hub` is the
    convenience path for named hub datasets.
    """

    def __init__(self, dataset: hf_datasets.Dataset, batch_size: int = 1000) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    @classmethod
    def from_hub(cls, name: str, split: str, batch_size: int = 1000) -> HFSource:
        try:
            import datasets
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "HFSource requires the 'datasets' package; install with: pip install 'crucible-data[hf]'"
            ) from exc
        return cls(datasets.load_dataset(name, split=split), batch_size=batch_size)

    def batches(self) -> Iterator[pa.Table]:
        for start in range(0, len(self.dataset), self.batch_size):
            chunk = self.dataset[start : start + self.batch_size]  # dict of columns
            yield pa.table(chunk)


_SUFFIX_TO_SOURCE: dict[str, Callable[..., Source]] = {
    ".parquet": ParquetSource,
    ".csv": CsvSource,
    ".jsonl": JsonlSource,
    ".json": JsonlSource,  # we only support newline-delimited JSON
}


def open_source(path: Path, fmt: str = "auto", batch_size: int = 1000) -> Source:
    """Pick a batch connector by explicit format or file suffix."""
    if fmt != "auto":
        by_name: dict[str, Callable[..., Source]] = {
            "parquet": ParquetSource,
            "csv": CsvSource,
            "jsonl": JsonlSource,
        }
        if fmt not in by_name:
            raise ValueError(f"unknown format {fmt!r}; expected one of {sorted(by_name)}")
        return by_name[fmt](path, batch_size=batch_size)
    source_cls = _SUFFIX_TO_SOURCE.get(path.suffix.lower())
    if source_cls is None:
        raise ValueError(f"cannot infer format from suffix {path.suffix!r}; pass fmt= explicitly")
    return source_cls(path, batch_size=batch_size)
