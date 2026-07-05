# Changelog

## Unreleased

### Added

- CLI bronze ingestion with `crucible ingest`.
- DuckDB-backed catalog inspection with `crucible sql` and `crucible catalog`.
- Offline smoke coverage through batch JSONL ingestion and the in-memory
  streaming path.
- Data contract, roadmap, and changelog docs for the shipped Phase 1 scope.
- Full local gate via `make gate`.
- Python 3.12 CI gate with coverage enforcement.

### Changed

- Project status now reflects Phase 1 bronze ingestion as shipped.
- CI runs the same full gate used locally.
