# Data Contracts

Crucible treats each data layer as a contract. The current shipped scope is
bronze ingestion; later phases will add the silver, gold, and quarantine
contracts without changing bronze semantics.

## Catalog Root

The catalog root defaults to `data/crucible` and is configurable with
`crucible ingest --root`, `crucible sql --root`, and `crucible catalog --root`.
Inside that root, each layer owns dataset directories:

```text
data/crucible/
  bronze/
    synth/
      part-<batch-hash-prefix>.parquet
      _ingest_log.jsonl
```

Dataset names are intentionally restricted because they become DuckDB view
names. A dataset must match `^[a-z][a-z0-9_]{0,62}$`.

## Bronze

Bronze is the raw landing layer. It is append-only at the dataset level and
immutable at the part level.

Part files:

- Stored as Parquet.
- Written through an atomic temporary-file rename.
- Named from the SHA-256 hash of canonical batch rows.
- Safe to republish by name, because the lander only uses content-addressed
  names.

Idempotency:

- Re-ingesting the same batch skips the already-landed part.
- Streaming redelivery after a crash is safe when the same consumer framing is
  replayed.
- Row-level duplicates across different batches are preserved. Deduplication is
  a later silver-stage concern.

Schema:

- The first non-empty batch establishes the dataset schema.
- Later batches may arrive with columns in a different order.
- Later batches may not add, remove, or change uncastable column types.
- All-null columns in the first batch are promoted to nullable strings so later
  non-null values can land.

Operational metadata:

- `_ingest_log.jsonl` records the batch hash, part name, row count, source name,
  Crucible version, and ingest timestamp for each landed part.
- The ingest log is not dataset identity. Dataset identity is the content of the
  Parquet parts.

## SQL Catalog

`Catalog.connect()` creates an in-memory DuckDB connection with one view per
non-empty dataset. View names are `<layer>_<dataset>`, for example
`bronze_synth`.

The `crucible sql` command prints query results as JSON rows. The
`crucible catalog` command prints per-layer dataset summaries with part and row
counts.

## Synthetic Corpus

The synthetic generator currently emits records with this schema:

| Field | Type | Contract |
|---|---|---|
| `id` | string | Stable record id, monotone in event time for a fixed config |
| `text` | string | Document text, including planted quality defects |
| `source` | string | Domain/source label used for mixture and drift experiments |
| `timestamp` | string | ISO 8601 UTC timestamp |
| `gt_kind` | string | Evaluation-only ground-truth defect label |
| `gt_dup_of` | string or null | Evaluation-only original id for duplicate records |

Pipeline stages must not use `gt_*` fields to make decisions. Those fields exist
so tests and later experiments can score stage output exactly.

## Planned Layers

Silver will contain schema-valid, quality-gated, deduplicated records plus
quarantine side outputs for rejected rows. Gold will contain curated mixtures
and sharding-ready datasets. Those contracts will be added when their phases
ship.
