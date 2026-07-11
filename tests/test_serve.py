import json
from pathlib import Path

import pytest

from crucible.dashboard import dashboard_data, render_dashboard
from crucible.observability import MetricsStore, StageMetric
from crucible.serve import create_app

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")


def test_metadata_api_empty_catalog_and_errors(tmp_path: Path) -> None:
    client = testclient.TestClient(create_app(tmp_path))
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/v1/catalog").json() == {}
    assert client.get("/v1/lineage").json() == {"datasets": [], "jobs": [], "edges": []}
    assert client.get("/v1/reports/quality/missing").status_code == 404
    assert client.get("/v1/reports/other/missing").status_code == 400
    assert client.get("/v1/runs?limit=0").status_code == 422


def test_metadata_api_exposes_reports_runs_and_metrics(tmp_path: Path) -> None:
    report = tmp_path / "reports" / "quality" / "synth.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"verdict": "promoted"}))
    run = tmp_path / "runs" / "r1" / "run.json"
    run.parent.mkdir(parents=True)
    run.write_text(json.dumps({"run_id": "r1", "status": "complete"}))
    MetricsStore(tmp_path).append(
        StageMetric("r1", "promote", "complete", "now", 1.0, output_rows=10)
    )

    client = testclient.TestClient(create_app(tmp_path))
    assert client.get("/v1/reports/quality/synth").json()["verdict"] == "promoted"
    assert client.get("/v1/runs").json()[0]["run_id"] == "r1"
    assert client.get("/v1/metrics?run_id=r1").json()[0]["output_rows"] == 10
    data = dashboard_data(tmp_path)
    assert data["runs"][0]["status"] == "complete"
    assert data["lineage_mermaid"] == "flowchart LR"


class _Tab:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class _Streamlit:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        def call(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            self.calls.append(name)
            if name == "tabs":
                return [_Tab(), _Tab(), _Tab()]
            return None

        return call


def test_dashboard_renders_all_sections(tmp_path: Path) -> None:
    streamlit = _Streamlit()
    render_dashboard(tmp_path, streamlit)
    assert streamlit.calls == [
        "set_page_config",
        "title",
        "caption",
        "tabs",
        "json",
        "subheader",
        "dataframe",
        "subheader",
        "dataframe",
        "code",
    ]
