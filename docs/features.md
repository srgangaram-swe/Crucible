# Feature layer

A minimal offline feature store whose one join is the only one safe for
training data: **point-in-time correct** attachment of the latest feature
row with `feature_ts <= spine_ts` (inclusive), implemented as a DuckDB
ASOF LEFT JOIN.

## Why PIT is the whole point

A naive "join the latest value" leaks the future into training rows and
silently inflates offline metrics — the classic feature-store failure.
Here that failure is *demonstrated* in the tests (a spine row between two
feature updates must get the earlier one; the naive join would return the
later) and guarded twice:

1. The ASOF join is correct by construction.
2. `assert_no_leakage` re-validates every joined frame — each attached
   feature timestamp must not postdate its spine row — and raises
   `LeakageError` otherwise. It runs automatically inside
   `point_in_time_join` and is exported for use on any external join.

## Offline/online parity

`get_latest(view, entity)` answers "what would the online store serve
right now". The parity test asserts it equals the PIT join evaluated at
t = ∞ — both stores answer from the same materialized truth, so a model
trained offline sees the same feature values it will be served online.

## Demo features

`source_rollup_features` computes cumulative per-source stats known *as
of* each record's timestamp (`docs_so_far`, `mean_words_so_far`) — the
kind of slowly-changing aggregate a mixture policy or curriculum would
consume, and cumulative by construction so it cannot peek forward. The
smoke test registers it from silver and PIT-joins a 100-row spine
(leak-checked) every run.

## Scope notes (honest)

- Views are single Parquet files with JSON metadata — no TTLs, no
  streaming refresh, no online serving infra; `get_latest` is a scan, not
  a KV store. The contract (PIT + parity) is what matters here, not scale.
- Entities and timestamps are strings (ISO-8601 UTC sorts correctly);
  DuckDB handles the comparison casts.
