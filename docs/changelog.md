# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

## [0.5.0] - 2026-07-06 — Phase 4: versioning & lineage

### Added

- `crucible.versioning`: content-addressed dataset manifests (sorted
  part:sha256 lines); version snapshots pinning (stage, config hash, input
  manifest hashes, code version, output hash), written automatically by
  promote and dedup; `verify_snapshot` detects tampering and stale pins.
- `crucible.lineage`: OpenLineage-inspired COMPLETE events appended by
  ingest/promote/dedup (blocked promotions emit too, with empty outputs);
  `LineageGraph` with upstream ancestry and Mermaid rendering.
- CLI: `crucible versions`, `crucible verify-snapshot`, `crucible lineage
  [--mermaid]`.
- Smoke stages 10-11: lineage/snapshot presence + verification, and a
  byte-identical rebuild of silver in a fresh catalog (the reproducibility
  contract, proven not claimed); docs/lineage.md; metadata-plane contract.

## [0.4.0] - 2026-07-05 — Phase 3: deduplication

### Added

- `crucible.dedup`: exact dedup (normalized-text hashing) + from-scratch
  seed-deterministic MinHash/banded-LSH near-dup detection with
  exact-Jaccard verification and union-find clustering; earliest record
  (smallest id) kept per cluster; datasketch as a config-switchable backend
  pinned to the same LSH banding (`dedup` extra, in dev deps so tested).
- `crucible.assay`: dedup scoring vs planted `gt_dup_of` including
  `fp_unlabeled_exact` (accidental template collisions counted, not
  forgiven) and a threshold sweep. Measured at threshold 0.5: exact recall
  1.0, near recall 0.7, precision 0.78, F1 0.79; precision 1.0 at 0.6.
- `Catalog.replace_dataset`: shared content-addressed rewriting of derived
  layers (bronze refuses); gate and dedup both use it.
- CLI: `crucible dedup`, `crucible score-dedup [--sweep]`;
  `configs/dedup_default.yaml`; smoke stage 9 with measured assertions;
  docs/dedup.md with the full measured tradeoff table.

### Fixed

- Cluster survivors are chosen by stable id order, not row position —
  content-hash part names make `Catalog.read` order arbitrary, which made
  naive keep-first keep copies instead of originals (caught by measurement:
  exact-dup recall was 0.75 before, 1.0 after).

## [0.3.0] - 2026-07-05 — Phase 2: quality gates

### Added

- `crucible.quality`: native Arrow rule engine (C4/RefinedWeb-inspired
  heuristics) with hit rates verified exactly against planted defects;
  bronze→silver promotion gate with quarantine (`reject_reasons`) and
  whole-promotion blocking above a reject-rate ceiling; PSI drift detection
  with JSON-serializable dataset profiles; JSON + Markdown gate reports.
- `crucible.assay`: ground-truth scoring of gate decisions (the only code
  allowed to read `gt_*`). Measured on the seed-42 smoke corpus: precision
  1.0, recall 1.0 with default rules; the opt-in `no_repeated_sentences`
  rule documented as a measured precision/keep-rate tradeoff.
- pandera bridge (`quality` extra, included in dev deps so it is tested):
  declarative second opinion on silver output.
- CLI: `crucible promote`, `crucible score-gate` (evaluation-only),
  `crucible drift`; `configs/quality_default.yaml`.
- Smoke now runs gate → quarantine → measured scoring → drift detection
  (~0.4s end to end); docs/quality.md; silver/quarantine data contracts.

## [0.2.0] - 2026-07-05 — Phase 1: ingestion + medallion storage

### Added

- `crucible.storage`: medallion Catalog (bronze/silver/gold/quarantine) with
  atomic, content-addressed Parquet parts and DuckDB views per dataset.
- `crucible.ingest`: Parquet/CSV/JSONL/HF batch connectors; content-addressed
  idempotent bronze landing with crash-resume; schema pinning with drift
  rejection and null-column promotion.
- Streaming path: Broker/Consumer protocols, in-memory fallback broker with
  bounded-backlog backpressure, Kafka adapter behind the `stream` extra,
  opt-in Kafka integration test (`CRUCIBLE_KAFKA_BOOTSTRAP`).
- CLI bronze ingestion with `crucible ingest` (`--via-stream` exercises the
  broker path).
- DuckDB-backed catalog inspection with `crucible sql` and `crucible catalog`.
- Offline smoke coverage through batch JSONL ingestion and the in-memory
  streaming path, including idempotency and cross-path equivalence checks.
- Data contract, roadmap, and changelog docs for the shipped Phase 1 scope.
- Full local gate via `make gate`; Python 3.12 CI gate with coverage
  enforcement (fail_under=95).
- Repo hygiene: CONTRIBUTING, SECURITY, CODEOWNERS, issue/PR templates,
  docker-compose profiles for the optional Kafka/MinIO path.
- `hf` extra for the Hugging Face datasets connector.

### Changed

- Project status now reflects Phase 1 bronze ingestion as shipped.
- CI runs the same full gate used locally.

## [0.1.0] - 2026-07-04 — Phase 0: scaffold

### Added

- Packaging (src layout, extras per subsystem), `crucible` CLI, YAML→pydantic
  configs, canonical content hashing.
- Deterministic synthetic corpus generator with evaluation-only ground-truth
  defect labels (`gt_kind`, `gt_dup_of`).
- ruff + black + mypy --strict + pre-commit; Makefile; GitHub Actions CI;
  offline smoke check.
