# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

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
