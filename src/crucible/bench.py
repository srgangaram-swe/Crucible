"""Throughput benchmarks: measured numbers or nothing.

Runs the full pipeline on a synthetic corpus of configurable size and
records wall-clock throughput per stage into a JSON file that carries
everything needed to interpret it: host info, package version, and the
exact config. Any performance number quoted in docs must come from one of
these files (see benchmarks/README.md); the trainer stage is included only
when torch is installed and the result says so either way.
"""

from __future__ import annotations

import json
import platform
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crucible import __version__
from crucible.dedup import DedupConfig, run_dedup
from crucible.ingest import JsonlSource, land
from crucible.quality import QualityConfig, run_gate
from crucible.shards import ShardConfig, ShardReader, build_shards
from crucible.storage import Catalog
from crucible.synth import SynthConfig, generate_corpus, write_jsonl


def _timed(started: float, units: int) -> dict[str, float]:
    elapsed = time.perf_counter() - started
    return {"elapsed_s": round(elapsed, 4), "per_second": round(units / elapsed, 1)}


def run_bench(
    n_docs: int = 5000,
    seq_len: int = 256,
    train_steps: int = 30,
    seed: int = 0,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Measure stage throughput end to end; returns (and writes) the report."""
    stages: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="crucible-bench-") as tmp:
        workdir = Path(tmp)
        corpus = workdir / "corpus.jsonl"
        records = generate_corpus(SynthConfig(seed=seed, n_docs=n_docs))
        write_jsonl(records, corpus)
        catalog = Catalog(workdir / "catalog")

        started = time.perf_counter()
        ingest_result = land(JsonlSource(corpus, batch_size=1000), catalog, "synth", "bench")
        stages["ingest_rows"] = _timed(started, ingest_result.rows_written)

        started = time.perf_counter()
        gate_result = run_gate(catalog, "synth", QualityConfig())
        stages["gate_rows"] = _timed(started, gate_result.input_rows)

        started = time.perf_counter()
        dedup_result = run_dedup(catalog, "synth", DedupConfig())
        stages["dedup_rows"] = _timed(started, dedup_result.input_rows)

        started = time.perf_counter()
        shard_result = build_shards(catalog, "synth", ShardConfig(seq_len=seq_len, seed=seed))
        stages["shard_build_tokens"] = _timed(started, shard_result.n_tokens)

        reader = ShardReader(catalog, "synth_shards", seed=seed)
        started = time.perf_counter()
        read_tokens = sum(len(seq) for seq in reader.iterate(epoch=0))
        stages["shard_read_tokens"] = _timed(started, read_tokens)

        try:
            from crucible.train import TrainConfig, train

            train_result = train(
                catalog,
                TrainConfig(shards_dataset="synth_shards", steps=train_steps, seed=seed),
            )
            stages["train"] = {
                "steps": train_result.steps,
                "tokens_per_second": train_result.tokens_per_second,
                "elapsed_s": train_result.elapsed_s,
                "device": train_result.device,
                "final_loss": train_result.final_loss,
            }
        except ImportError:
            stages["train"] = "skipped: torch not installed"

    report: dict[str, Any] = {
        "kind": "crucible-bench",
        "created_at": datetime.now(UTC).isoformat(),
        "crucible_version": __version__,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "config": {"n_docs": n_docs, "seq_len": seq_len, "train_steps": train_steps, "seed": seed},
        "corpus": {
            "rows": n_docs,
            "silver_rows": dedup_result.kept_rows,
            "sequences": shard_result.n_sequences,
            "tokens": shard_result.n_tokens,
        },
        "stages": stages,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        (out_dir / f"bench-{stamp}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
