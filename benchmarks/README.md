# Benchmarks

`make bench` (or `crucible bench`) runs the full pipeline — synth → ingest →
gate → dedup → shard build → shard read → a few trainer steps — on a
configurable corpus and writes per-stage wall-clock throughput to
`results/bench-<timestamp>.json`, tagged with host info, package version,
and the exact config.

Rules:

- Any performance number quoted in docs or the README must come from a
  committed result file here; no exceptions.
- Results are per-host snapshots, not comparable claims — read the `host`
  block before comparing two files.
- The trainer stage records `skipped: torch not installed` rather than
  omitting itself silently.
