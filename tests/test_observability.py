from pathlib import Path

import pytest

from crucible.observability import MetricsStore, StageTimer


def test_timer_records_success_and_throughput(tmp_path: Path) -> None:
    store = MetricsStore(tmp_path)
    with StageTimer(store, "run-1", "promote") as timer:
        timer.input_rows = 20
        timer.output_rows = 10
    [metric] = store.list()
    assert metric["status"] == "complete"
    assert metric["input_rows"] == 20
    assert metric["throughput_rows_per_second"] > 0


def test_timer_records_failure_and_store_filters(tmp_path: Path) -> None:
    store = MetricsStore(tmp_path)
    with (
        pytest.raises(ValueError, match="bad input"),
        StageTimer(store, "run-failed", "gate", attempt=2),
    ):
        raise ValueError("bad input")
    store.path.write_text(store.path.read_text() + "not-json\n")
    [metric] = store.list(run_id="run-failed")
    assert metric["status"] == "failed"
    assert metric["attempt"] == 2
    assert metric["error"] == "bad input"
    assert store.list(run_id="other") == []
