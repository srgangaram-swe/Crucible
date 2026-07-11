import json
from pathlib import Path

import pytest

from crucible.dedup import DedupConfig
from crucible.ingest import land, open_source
from crucible.orchestrate import DagError, DagRunner, RunContext, Task, run_pipeline
from crucible.quality import QualityConfig
from crucible.storage import Catalog, Layer
from crucible.synth import SynthConfig, generate_corpus, write_jsonl


def test_dag_validates_dependencies_and_cycles(tmp_path: Path) -> None:
    def noop(context: RunContext) -> None:
        pass

    runner = DagRunner(tmp_path)
    with pytest.raises(DagError, match="unknown"):
        runner.run([Task("a", noop, ("missing",))], "data")
    with pytest.raises(DagError, match="cycle"):
        runner.run([Task("a", noop, ("b",)), Task("b", noop, ("a",))], "data")


def test_dag_orders_retries_and_persists_failure(tmp_path: Path) -> None:
    calls: list[str] = []

    def flaky(context: RunContext) -> dict[str, int]:
        calls.append("flaky")
        if len(calls) == 1:
            raise RuntimeError("transient")
        return {"output_rows": 3}

    def final(context: RunContext) -> None:
        calls.append("final")

    result = DagRunner(tmp_path).run(
        [Task("final", final, ("flaky",)), Task("flaky", flaky, retries=1)],
        "data",
        run_id="known-run",
    )
    assert calls == ["flaky", "flaky", "final"]
    assert result.status == "complete"
    assert result.tasks[0]["attempts"] == 2
    assert len((tmp_path / "metrics" / "stages.jsonl").read_text().splitlines()) == 3

    def fail(context: RunContext) -> None:
        raise ValueError("permanent")

    with pytest.raises(DagError, match="permanent"):
        DagRunner(tmp_path).run([Task("fail", fail)], "data", run_id="failed-run")
    record = json.loads((tmp_path / "runs" / "failed-run" / "run.json").read_text())
    assert record["status"] == "failed"


def test_pipeline_is_idempotent_and_forceable(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    write_jsonl(generate_corpus(SynthConfig(n_docs=60, seed=4)), raw)
    catalog = Catalog(tmp_path / "catalog")
    land(open_source(raw), catalog, "synth", raw.name)

    first = run_pipeline(catalog.root, "synth", QualityConfig(), DedupConfig())
    second = run_pipeline(catalog.root, "synth", QualityConfig(), DedupConfig())
    forced = run_pipeline(catalog.root, "synth", QualityConfig(), DedupConfig(), force=True)
    assert first.status == "complete"
    assert second.status == "skipped"
    assert second.run_id == first.run_id
    assert forced.status == "complete" and forced.run_id != first.run_id
    assert catalog.row_count(Layer.SILVER, "synth") > 0
