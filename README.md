# Crucible

*A local-first data refinery and data-centric research platform for distributed training.*

[![CI](https://github.com/srgangaram-swe/Crucible/actions/workflows/ci.yml/badge.svg)](https://github.com/srgangaram-swe/Crucible/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

## Why this exists

Model quality at scale is increasingly determined by **what you train on, not just how you
train**. Yet most open-source ML tooling focuses on the model side; the training-data side —
ingestion, validation, deduplication, versioning, mixing, sharding — is where large labs invest
heavily and where public, end-to-end reference implementations are rare.

Crucible is that reference implementation, deliberately built at laptop scale:

1. **A training-data platform**: batch + streaming ingestion into bronze/silver/gold Parquet
   layers, statistical quality gates with quarantine, exact + MinHash/LSH near-duplicate
   removal, content-addressed dataset versions with lineage, a point-in-time-correct feature
   layer, and a builder that emits streaming-friendly training shards.
2. **A data-centric research harness**: seed-controlled, YAML-configured experiments that use
   the platform to measure how deduplication thresholds, domain-mixture ratios, quality gating,
   and data scale affect a small transformer trained on the resulting shards.

Everything runs **offline on a laptop CPU with zero external services** (in-memory broker,
DuckDB, local filesystem), and scales up along documented switches (Kafka/Redpanda, MinIO/S3,
multi-GPU DDP/FSDP).

The name: a crucible is the vessel where raw ore is refined — and "put through the crucible"
means rigorously tested. Both meanings apply. Raw data goes through bronze → silver → gold
refinement, and the `assay` experiment harness measures the purity of what comes out.

## Status

**Phase 0 of 8** — scaffold, tooling, CI, synthetic data generator, smoke path. The roadmap
below is honest about what exists versus what is planned.

| Phase | Subsystem | Status |
|---|---|---|
| 0 | Scaffold, CI, CLI, synthetic corpus generator, smoke test | ✅ done |
| 1 | Ingestion (batch + streaming w/ in-memory fallback) → bronze/silver/gold + DuckDB | 🔜 next |
| 2 | Quality gates, quarantine, drift detection, reports | planned |
| 3 | Exact + MinHash/LSH deduplication with measured rates | planned |
| 4 | Content-addressed versioning, manifests, lineage graph | planned |
| 5 | Feature layer with point-in-time joins + leakage guards | planned |
| 6 | Training-shard builder + DDP/FSDP reference trainer | planned |
| 7 | Orchestration DAG, FastAPI service, Streamlit dashboard | planned |
| 8 | Research harness + capstone data-centric study + docs | planned |

## Quickstart

```bash
git clone https://github.com/srgangaram-swe/Crucible && cd Crucible
make install          # creates .venv and installs crucible + dev tools
make smoke            # end-to-end offline check (~seconds on CPU)
make test lint type   # full quality gate
```

Generate a synthetic corpus with known, labeled defects (duplicates, near-duplicates, junk,
synthetic PII) — the substrate for every test and experiment in the repo:

```bash
.venv/bin/crucible synth --config configs/synth_small.yaml --out data/raw/synth
```

Every record carries evaluation-only ground truth (`gt_kind`, `gt_dup_of`), so downstream
stages can report *measured* precision/recall for dedup and quality gates instead of vibes.

## Design principles

- **Local-first, then scale.** The default path needs no Docker, no network, no GPU. Optional
  extras (`.[stream]`, `.[dedup]`, `.[train]`, `.[serve]`) light up each subsystem.
- **Reproducible by construction.** Every artifact is a pure function of a YAML config and a
  seed; datasets are identified by content hash.
- **Measured, not claimed.** Reported numbers come from runs you can re-execute; illustrative
  numbers are labeled as such. See [docs/limitations.md](docs/limitations.md).

## Repository layout

```
src/crucible/       the package (CLI: `crucible`)
tests/              pytest suite (unit + integration; `make test`)
configs/            YAML configs — every pipeline/experiment is driven by one
docs/               architecture, data contracts, lineage, experiments, limitations
```

## License

MIT — see [LICENSE](LICENSE).
