"""End-to-end smoke check: proves the offline path works on this machine.

Phase 0 scope: generate the synthetic corpus twice, verify byte-for-byte
determinism, verify ground-truth invariants, round-trip through Parquet,
and query it back with DuckDB. Later phases extend this to the full
ingest -> validate -> dedup -> version -> shard -> train pipeline; ``make
smoke`` must always exercise everything that exists so far, on CPU, with
no external services, in about a minute.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import duckdb

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

    report = generation_report(_SMOKE_CONFIG, records)
    return {
        "ok": True,
        "phase": 0,
        "checks_passed": checks,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "corpus_sha256": digest,
        "by_source_via_duckdb": by_source,
        "generation": report,
    }
