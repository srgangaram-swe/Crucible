from pathlib import Path

from crucible.bench import run_bench


def test_bench_runs_tiny_and_writes_report(tmp_path: Path) -> None:
    report = run_bench(n_docs=120, seq_len=64, train_steps=2, out_dir=tmp_path)
    assert report["kind"] == "crucible-bench"
    assert report["config"]["n_docs"] == 120
    for stage in (
        "ingest_rows",
        "gate_rows",
        "dedup_rows",
        "shard_build_tokens",
        "shard_read_tokens",
    ):
        assert report["stages"][stage]["per_second"] > 0
    assert report["host"]["python"]
    files = list(tmp_path.glob("bench-*.json"))
    assert len(files) == 1
