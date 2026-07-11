"""Streamlit dashboard presentation kept separate from metadata collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crucible.lineage import LineageGraph
from crucible.observability import MetricsStore
from crucible.orchestrate import list_runs
from crucible.storage import Catalog


def dashboard_data(root: Path) -> dict[str, Any]:
    graph = LineageGraph.from_root(root)
    return {
        "catalog": Catalog(root).summary(),
        "runs": list_runs(root),
        "metrics": MetricsStore(root).list(),
        "lineage_mermaid": graph.to_mermaid(),
    }


def render_dashboard(root: Path, streamlit: Any) -> None:
    data = dashboard_data(root)
    streamlit.set_page_config(page_title="Crucible", page_icon="🔥", layout="wide")
    streamlit.title("Crucible control plane")
    streamlit.caption(f"Local metadata from {Path(root).resolve()}")
    catalog_tab, runs_tab, lineage_tab = streamlit.tabs(["Catalog", "Runs & metrics", "Lineage"])
    with catalog_tab:
        streamlit.json(data["catalog"])
    with runs_tab:
        streamlit.subheader("Pipeline runs")
        streamlit.dataframe(data["runs"], use_container_width=True)
        streamlit.subheader("Stage metrics")
        streamlit.dataframe(data["metrics"], use_container_width=True)
    with lineage_tab:
        streamlit.code(data["lineage_mermaid"], language="mermaid")


def main() -> None:
    try:
        import streamlit
    except ImportError as exc:
        raise RuntimeError("dashboard requires `pip install crucible-data[serve]`") from exc
    render_dashboard(Path("data/crucible"), streamlit)


if __name__ == "__main__":
    main()
