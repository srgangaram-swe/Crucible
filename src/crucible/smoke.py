"""End-to-end smoke check: proves the offline path works on this machine.

Current scope (Phase 1): generate the synthetic corpus, verify determinism
and ground-truth invariants, then land it in bronze through BOTH ingestion
paths — batch JSONL and the in-memory streaming broker — and verify
idempotency and cross-path equivalence via DuckDB. Later phases extend
this to validate -> dedup -> version -> shard -> train; ``make smoke``
must always exercise everything that exists so far, on CPU, with no
external services, in about a minute.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import duckdb

from crucible.ingest import InMemoryBroker, JsonlSource, StreamSource, land, replay_jsonl
from crucible.storage import Catalog
from crucible.synth import (
    SynthConfig,
    corpus_sha256,
    generate_corpus,
    generation_report,
    write_jsonl,
    write_parquet,
)

_SMOKE_CONFIG = SynthConfig(seed=42, n_docs=400)


class SmokeFailure(Exception):
    """A smoke check failed; message says which."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def run_smoke(workdir: Path | None = None) -> dict[str, Any]:
    """Run all smoke checks; return a report dict. Raises SmokeFailure."""
    started = time.perf_counter()
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="crucible-smoke-")
        workdir = Path(tmp.name)
    workdir.mkdir(parents=True, exist_ok=True)

    checks: list[str] = []

    # 1. Determinism: same config -> byte-identical corpus.
    records = generate_corpus(_SMOKE_CONFIG)
    digest = corpus_sha256(records)
    digest_again = corpus_sha256(generate_corpus(_SMOKE_CONFIG))
    _check(digest == digest_again, "generation is not deterministic for a fixed seed")
    checks.append("determinism")

    # 2. Ground-truth invariants.
    _check(len(records) == _SMOKE_CONFIG.n_docs, "record count != n_docs")
    by_id = {record.id: record for record in records}
    _check(len(by_id) == len(records), "record ids are not unique")
    for record in records:
        if record.gt_dup_of is not None:
            _check(record.gt_dup_of in by_id, f"{record.id}: dangling gt_dup_of")
            if record.gt_kind == "exact_dup":
                _check(
                    record.text == by_id[record.gt_dup_of].text,
                    f"{record.id}: exact_dup text differs from original",
                )
    timestamps = [record.timestamp for record in records]
    _check(timestamps == sorted(timestamps), "records are not time-ordered")
    checks.append("ground_truth_invariants")

    # 3. Parquet + JSONL round trip, queried back via DuckDB.
    parquet_path = workdir / "corpus.parquet"
    jsonl_path = workdir / "corpus.jsonl"
    write_parquet(records, parquet_path)
    write_jsonl(records, jsonl_path)
    with duckdb.connect() as conn:
        row = conn.execute(
            "SELECT count(*), count(DISTINCT id) FROM read_parquet(?)",
            [str(parquet_path)],
        ).fetchone()
        if row is None:
            raise SmokeFailure("duckdb count query returned no rows")
        n_rows, n_ids = row
        by_source = dict(
            conn.execute(
                "SELECT source, count(*) FROM read_parquet(?) GROUP BY source ORDER BY source",
                [str(parquet_path)],
            ).fetchall()
        )
    _check(n_rows == len(records), "duckdb row count mismatch after parquet round trip")
    _check(n_ids == len(records), "duckdb distinct-id count mismatch")
    checks.append("parquet_duckdb_roundtrip")

    # 4. Batch ingest to bronze; re-ingest must be a no-op.
    catalog = Catalog(workdir / "catalog")
    first = land(JsonlSource(jsonl_path, batch_size=97), catalog, "synth", "smoke-batch")
    _check(first.rows_written == len(records), "batch ingest lost or duplicated rows")
    again = land(JsonlSource(jsonl_path, batch_size=97), catalog, "synth", "smoke-batch")
    _check(
        again.parts_written == 0 and again.rows_skipped == len(records),
        "re-ingest was not idempotent",
    )
    checks.append("bronze_batch_ingest_idempotent")

    # 5. Streaming fallback path lands the same content.
    broker = InMemoryBroker()
    published = replay_jsonl(jsonl_path, broker, "synth")
    _check(published == len(records), "replay did not publish every record")
    stream = StreamSource(broker, "synth", batch_size=113, poll_timeout=0.05)
    streamed = land(stream, catalog, "synth_stream", "smoke-stream")
    _check(streamed.rows_written == len(records), "stream ingest lost or duplicated rows")
    diff = catalog.query(
        "SELECT count(*) AS n FROM ("
        "  (SELECT id FROM bronze_synth EXCEPT SELECT id FROM bronze_synth_stream)"
        "  UNION ALL"
        "  (SELECT id FROM bronze_synth_stream EXCEPT SELECT id FROM bronze_synth))"
    )
    _check(diff == [{"n": 0}], "batch and stream paths landed different content")
    checks.append("bronze_stream_fallback_equivalent")

    report = generation_report(_SMOKE_CONFIG, records)
    return {
        "ok": True,
        "phase": 1,
        "checks_passed": checks,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "corpus_sha256": digest,
        "by_source_via_duckdb": by_source,
        "bronze": catalog.summary().get("bronze", {}),
        "generation": report,
    }
