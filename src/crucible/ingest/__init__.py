"""Ingestion: batch and streaming sources landing idempotently in bronze.

Public API re-exported here; see :mod:`crucible.ingest.sources` (batch
connectors) and :mod:`crucible.ingest.land` (idempotent bronze landing).
"""

from crucible.ingest.land import IngestError, IngestResult, land
from crucible.ingest.sources import (
    CsvSource,
    HFSource,
    JsonlSource,
    ParquetSource,
    Source,
    open_source,
)

__all__ = [
    "CsvSource",
    "HFSource",
    "IngestError",
    "IngestResult",
    "JsonlSource",
    "ParquetSource",
    "Source",
    "land",
    "open_source",
]
