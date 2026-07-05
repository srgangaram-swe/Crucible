# Data Contracts

Crucible treats each data layer as a contract. The current shipped scope is
bronze ingestion and the bronze→silver quality gate (with quarantine); later
phases will add the gold contract without changing existing semantics.

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

## Silver

Silver contains exactly the bronze records that passed every enabled quality
rule. It is a **derived, rebuildable** layer:

- Promotion is a pure function of (bronze content, `QualityConfig`); re-running
  `crucible promote` rebuilds `silver/<dataset>` and `quarantine/<dataset>`
  from scratch and produces identical content-addressed parts.
- Schema is bronze's schema, unchanged. `gt_*` evaluation columns ride along
  untouched; the gate never reads them.
- If the reject rate exceeds `max_reject_rate`, **nothing** is promoted and the
  verdict is `blocked` — a systematically broken source fails loudly rather
  than leaking a plausible-looking silver dataset.
- After `crucible dedup`, silver is also deduplicated: exact duplicates
  (normalized-text hash) and verified near-duplicates (MinHash/LSH, Jaccard ≥
  threshold) are removed, keeping the smallest-id record per cluster. Dedup
  is likewise a pure function of (its input, config); the removed ids are
  recorded in `reports/dedup/<dataset>.json`. Pipeline order matters:
  promote, then dedup (re-promoting resurrects duplicates until the Phase 7
  DAG encodes the ordering).

## Quarantine

Quarantine holds the records the gate rejected, with one extra column:

| Field | Type | Contract |
|---|---|---|
| `reject_reasons` | string | `\|`-joined names of every rule the record failed |

Quarantine is diagnostic, not a dead letter queue: records are kept verbatim
(including planted PII — see limitations.md; production systems would redact)
so gate decisions can be audited and scored. `crucible score-gate` compares
quarantine membership against the planted `gt_*` labels and reports exact
precision/recall — on the smoke corpus, both are 1.0 with the default rules.

## Quality Reports

Each `crucible promote` writes `<root>/reports/quality/<dataset>.json` (full
config + result, machine-readable) and `.md` (human summary). Reports describe
the current promotion only; historical gate runs are reconstructable from
config + bronze.

## Gold (planned)

Gold will contain curated mixtures and sharding-ready datasets. Its contract
will be added when Phase 4+ ships.
