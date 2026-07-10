# Training shards

Shards are the gold layer: fixed-length packed token sequences stored as
content-addressed Parquet parts, so every existing contract — manifests,
version snapshots, lineage, DuckDB views (`gold_synth_shards`), and
byte-identical rebuilds — applies to training data with zero new machinery.
WebDataset tars or MosaicML MDS solve the same problem; Parquet keeps one
format across the whole refinery while `ShardReader` supplies the streaming
semantics.

## Building

`build_shards(catalog, "synth", ShardConfig(seq_len=256, seed=0))`:

1. Read silver, order rows by **id** (never by row position — part read
   order is arbitrary), then apply a seeded permutation.
2. Byte-level tokenize (vocab 256 + PAD/BOS/EOS = 259; zero deps, no
   network, nothing to version) and concatenate as `tokens + [EOS]`.
3. Slice the stream into `seq_len + 1`-token sequences (the trainer takes
   `x = t[:-1], y = t[1:]`), dropping the trailing partial.
4. Publish via `Catalog.replace_dataset` into `gold/<dataset>_shards`,
   emit the lineage event (`silver/synth → shard:synth → gold/synth_shards`)
   and write a `shard` version snapshot.

Determinism is tested: two fresh catalogs fed the same silver + config
produce byte-identical gold manifests; changing the seed changes them.

## Reading (streaming, shuffled, resumable)

`ShardReader(catalog, "synth_shards", seed, shuffle_buffer).iterate(epoch)`
streams sequences through a seeded shuffle buffer (per-epoch RNG streams, so
epochs reshuffle deterministically). The iterator checkpoint is just
`{"epoch", "consumed"}`; resuming replays and discards the first `consumed`
draws from the identical RNG stream, so **resume is exact** — asserted by
tests and by the smoke check `gold_shards_resumable`
(`head + resumed tail == full epoch`, element for element).

## Scope notes (honest)

- Replay-to-resume is O(consumed); at web scale you would checkpoint
  buffer contents or skip whole shard blocks. Right fidelity here: the
  contract (exactness) is tested, the optimization is documented.
- Byte-level tokenization inflates sequence counts vs subword; swapping a
  trained tokenizer in is a config-level change the Phase 8 harness can
  study (unique-token effects interact with dedup).
- The shuffle buffer bounds memory but limits shuffle radius to the buffer
  size — standard streaming tradeoff (same as WebDataset/TF Data).
