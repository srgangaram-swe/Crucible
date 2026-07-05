"""End-to-end smoke check: proves the offline path works on this machine.

Current scope (Phase 2): generate the synthetic corpus, verify determinism
and ground-truth invariants, land it in bronze through BOTH ingestion paths
(batch JSONL and the in-memory streaming broker) with idempotency and
cross-path equivalence checks, then run the quality gate to silver +
quarantine, score it against the planted ground truth (recall must be
perfect, precision near-perfect — these are measured, not asserted hopes),
and verify PSI drift detection fires on a skewed mixture but not on
identically-distributed data. Later phases extend this to dedup ->
version -> shard -> train; ``make smoke`` must always exercise everything
that exists so far, on CPU, with no external services, in about a minute.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import duckdb

from crucible.assay import score_dedup, score_gate
from crucible.dedup import DedupConfig, run_dedup
from crucible.dedup.pipeline import write_dedup_report
from crucible.ingest import InMemoryBroker, JsonlSource, StreamSource, land, replay_jsonl
from crucible.quality import QualityConfig, drift_report, run_gate, write_report
from crucible.quality.drift import profile_table
from crucible.storage import Catalog, Layer
from crucible.synth import (
    SynthConfig,
    corpus_sha256,
    generate_corpus,
    generation_report,
    to_table,
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

    # 6. Quality gate: bronze -> silver + quarantine, with reports.
    gate_cfg = QualityConfig()
    gate_result = run_gate(catalog, "synth", gate_cfg)
    _check(gate_result.verdict == "promoted", "quality gate blocked the smoke corpus")
    _check(
        gate_result.promoted_rows + gate_result.quarantined_rows == len(records),
        "gate lost records between silver and quarantine",
    )
    _check(gate_result.quarantined_rows > 0, "gate quarantined nothing despite planted junk")
    write_report(gate_result, gate_cfg, catalog.root)
    checks.append("quality_gate_promoted")

    # 7. Measured gate quality vs planted ground truth (evaluation-only read).
    quarantined_ids = set(catalog.read(Layer.QUARANTINE, "synth").column("id").to_pylist())
    score = score_gate(catalog.read(Layer.BRONZE, "synth"), quarantined_ids)
    _check(score.recall == 1.0, f"gate missed planted defects (recall={score.recall})")
    _check(score.precision >= 0.95, f"gate over-quarantined (precision={score.precision})")
    checks.append("gate_precision_recall_measured")

    # 8. Drift: a skewed mixture must register, an identical one must not.
    skewed = generate_corpus(
        SynthConfig(
            seed=43,
            n_docs=200,
            domain_weights={"news": 0.05, "forum_qa": 0.05, "code": 0.85, "recipes": 0.05},
        )
    )
    bronze_profile = profile_table(catalog.read(Layer.BRONZE, "synth"))
    drift_vs_skewed = drift_report(bronze_profile, profile_table(to_table(skewed)))
    drift_vs_self = drift_report(bronze_profile, bronze_profile)
    _check(drift_vs_skewed["source_verdict"] == "major", "drift missed a skewed mixture")
    _check(drift_vs_self["verdict"] == "none", "drift false-alarmed on identical data")
    checks.append("drift_detection")

    # 9. Dedup silver; scored against planted duplicates (evaluation-only).
    silver_pre = catalog.read(Layer.SILVER, "synth")
    dedup_result = run_dedup(catalog, "synth", DedupConfig())
    write_dedup_report(dedup_result, catalog.root)
    _check(
        dedup_result.kept_rows + len(dedup_result.removed_ids) == silver_pre.num_rows,
        "dedup lost or duplicated rows",
    )
    dedup_score = score_dedup(silver_pre, set(dedup_result.removed_ids))
    _check(
        dedup_score.recall_by_kind.get("exact_dup") == 1.0,
        f"exact duplicates escaped dedup ({dedup_score.recall_by_kind})",
    )
    _check(dedup_score.f1 >= 0.75, f"dedup F1 regressed ({dedup_score.f1})")
    checks.append("silver_dedup_measured")

    report = generation_report(_SMOKE_CONFIG, records)
    return {
        "ok": True,
        "phase": 3,
        "checks_passed": checks,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "corpus_sha256": digest,
        "by_source_via_duckdb": by_source,
        "bronze": catalog.summary().get("bronze", {}),
        "quality": {
            "gate": gate_result.as_dict(),
            "score_vs_ground_truth": score.as_dict(),
            "drift_vs_skewed": drift_vs_skewed,
        },
        "dedup": {
            "kept_rows": dedup_result.kept_rows,
            "removed_exact": dedup_result.removed_exact,
            "removed_near": dedup_result.removed_near,
            "n_clusters": dedup_result.n_clusters,
            "score_vs_ground_truth": dedup_score.as_dict(),
        },
        "generation": report,
    }
