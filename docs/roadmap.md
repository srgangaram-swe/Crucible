# Roadmap

This roadmap is a living TODO list. A phase is marked done only when the CLI,
tests, docs, and smoke path for that phase are present.

## Done

### Phase 0: Scaffold

- Python package with typed source layout.
- CLI shell with deterministic synthetic corpus generation.
- Strict lint, formatting, type-checking, and pytest setup.
- Offline smoke command.

### Phase 1: Bronze Ingestion

- Batch connectors for JSONL, CSV, and Parquet.
- Streaming source adapter with in-memory broker fallback.
- Optional Kafka adapter behind the same broker protocol.
- Content-addressed, idempotent landing into bronze Parquet parts.
- DuckDB catalog views and JSON CLI output for SQL and catalog summaries.
- Smoke path that proves batch ingest, stream ingest, idempotency, and
  cross-path equivalence through bronze.

### Phase 2: Quality Gates

- Native Arrow rule engine (C4/RefinedWeb-style heuristics) with exact
  hit-rate verification against planted defects.
- Bronze→silver promotion as a pure function of (bronze, config); quarantine
  with `reject_reasons`; whole-promotion blocking above a reject-rate ceiling.
- PSI drift detection over source mix and length distribution.
- `crucible promote`, `crucible score-gate` (evaluation-only), `crucible drift`.
- Measured on the seed-42 corpus: precision 1.0, recall 1.0; the opt-in
  repeated-sentences rule documented as a precision/keep-rate tradeoff.
- pandera bridge as a declarative second opinion (quality extra).

### Phase 3: Deduplication

- Exact removal via normalized-text hashing; near-dup via from-scratch
  seed-deterministic MinHash + banded LSH with exact-Jaccard verification;
  union-find clusters keeping the smallest-id (earliest) record.
- datasketch as a config-switchable backend pinned to the same LSH banding.
- `crucible dedup` and `crucible score-dedup [--sweep ...]`; measured on the
  seed-42 corpus: exact recall 1.0 at every threshold, F1 0.79 at 0.5,
  precision 1.0 at 0.6 — full curve in docs/dedup.md.

### Phase 4: Versioning and Lineage

- Content-addressed dataset manifests (sorted part:sha256 lines).
- Snapshots pinning (stage, config hash, input hashes, code version, output
  hash), written automatically by promote/dedup; `crucible versions` and
  `crucible verify-snapshot` (stale pins fail loudly).
- OpenLineage-style events from every stage; `crucible lineage` graph with
  upstream ancestry and Mermaid rendering.
- Smoke-proven byte-identical rebuild of silver in a fresh catalog.

## Next

### Phase 5: Feature Layer

- Point-in-time joins.
- Leakage guards.
- Feature materialization metadata.

### Phase 6: Training Shards and Reference Trainer

- Tokenization and deterministic shard building.
- Streaming-friendly shard reader.
- Small CPU trainer first, with DDP/FSDP scale-up paths.

### Phase 7: Orchestration and Serving

- Idempotent DAG runner.
- FastAPI metadata service.
- Streamlit dashboard for catalog, quality, lineage, and run reports.

### Phase 8: Assay Research Harness

- Dedup, quality, mixture, and data-scaling ablations.
- Reproducible capstone study with configs and measured outputs.
