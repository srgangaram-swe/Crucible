# Limitations and honest scope

This project is an **engineering-scale reference implementation**, not a production system or
a large-scale study. Specifically:

- **Scale.** Default corpora are thousands-to-millions of synthetic records, not web-scale.
  Distributed paths (DDP/FSDP, Kafka, object storage) are real code but validated on small
  clusters at most; throughput numbers are measured on the hardware stated next to them.
- **Data.** The primary corpus is synthetic (template-grammar text with planted, labeled
  defects) plus small public datasets. Synthetic data makes defect metrics exact, but absolute
  numbers do not transfer to natural corpora; treat trends, not magnitudes, as the finding.
- **Models.** The reference trainer uses a deliberately small transformer. Research results
  are about *data* effects at small scale; extrapolation to frontier scale is not claimed.
- **Phase 8 evaluator.** The committed ablations use a Laplace-smoothed byte-bigram proxy
  model to keep multi-seed studies deterministic, offline, and CI-sized. Its held-out
  cross-entropy is a real downstream measurement but is not a substitute for transformer
  pretraining. Three-seed bootstrap intervals describe these runs; they do not establish
  population-level uncertainty or external validity.
- **Measured vs illustrative numbers.** Any number in docs or reports is either (a) produced
  by a committed, seed-controlled run whose config is referenced next to it, or (b) explicitly
  labeled "illustrative". Nothing in between — benchmark numbers are never fabricated.
- **Gate scores are synthetic-corpus scores.** The measured precision/recall of the quality
  gate (1.0/1.0 with default rules) is against *planted, template-generated* defects that the
  rules were designed to detect. Real web junk is adversarial and long-tailed; these rules
  would need re-tuning and the scores would not transfer. What does transfer is the
  methodology: labeled defects, blind pipeline, exact scoring.
- **PII is quarantined, not redacted.** Quarantine keeps rejected records verbatim so gate
  decisions can be audited and scored. A production system would redact or tokenize PII
  spans instead of parking them readable on disk; all planted PII here is synthetic
  (`user123@example.com`, 555 numbers).
- **Near-dup recall tops out below 1.0 by design.** Planted near-duplicates carry 8-15%
  token edits, placing their Jaccard band deliberately astride usable thresholds — at the
  F1-optimal threshold, near-dup recall is 0.7 (measured). A generator whose defects were
  all trivially detectable would make every reported metric meaningless. Some dedup "false
  positives" are actually unlabeled accidental template collisions; these are counted
  separately (`fp_unlabeled_exact`) instead of being forgiven.
- **Lineage is locally persisted and read-only served.** Events are OpenLineage-inspired JSONL in the
  catalog root, folded into a graph at read time; they are not validated against the
  OpenLineage JSON schema. The Phase 7 API/dashboard expose local metadata but are not a
  multi-tenant control plane (the emitter could target Marquez later). Snapshot verification
  checks content hashes, not a cryptographic audit trail.
- **Orchestration is single-process.** The DAG validates ordering, retries bounded failures,
  persists runs, and skips completed input/config fingerprints. It is not a distributed
  scheduler and has no leases, backfill calendar, or cross-host worker queue.
- **Re-promotion is not concurrency-safe.** `run_gate` clears and rebuilds derived layers;
  two concurrent promotions of the same dataset can interleave. Single-writer-per-dataset is
  assumed throughout (fine for the local DAG runner; an object-store deployment would need a
  lease or a versioned-pointer swap).

*(This file is updated each phase as real constraints are discovered.)*
