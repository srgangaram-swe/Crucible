"""Ingestion: batch and streaming sources landing idempotently in bronze.

Public API re-exported here; see :mod:`crucible.ingest.sources` (batch
connectors), :mod:`crucible.ingest.stream` (brokers + streaming source),
and :mod:`crucible.ingest.land` (idempotent bronze landing).
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
from crucible.ingest.stream import (
    Broker,
    BrokerBackpressure,
    Consumer,
    InMemoryBroker,
    KafkaBroker,
    StreamSource,
    replay_jsonl,
)

__all__ = [
    "Broker",
    "BrokerBackpressure",
    "Consumer",
    "CsvSource",
    "HFSource",
    "InMemoryBroker",
    "IngestError",
    "IngestResult",
    "JsonlSource",
    "KafkaBroker",
    "ParquetSource",
    "Source",
    "StreamSource",
    "land",
    "open_source",
    "replay_jsonl",
]
