"""Quality report artifacts: machine-readable JSON plus human Markdown.

Reports live under ``<root>/reports/quality/<dataset>.{json,md}`` and are
overwritten on each gate run — the report describes the *current* silver
promotion, and older gate runs are reconstructable from config + bronze
(which is the point of pure-function promotion).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crucible.quality.gate import GateResult, QualityConfig


def render_markdown(result: GateResult, cfg: QualityConfig) -> str:
    lines = [
        f"# Quality gate report: `{result.dataset}`",
        "",
        f"**Verdict: {result.verdict}** — {result.promoted_rows}/{result.input_rows} promoted, "
        f"{result.quarantined_rows} quarantined ({result.reject_rate:.1%}), "
        f"in {result.elapsed_s}s.",
        "",
        f"Rules: {', '.join(f'`{r}`' for r in cfg.rules)} "
        f"(min_words={cfg.min_words}, marker_threshold={cfg.boilerplate_marker_threshold}, "
        f"max_reject_rate={cfg.max_reject_rate:.0%})",
        "",
        "| Reject reason | Records |",
        "|---|---|",
    ]
    lines.extend(f"| `{reason}` | {count} |" for reason, count in result.reasons_histogram.items())
    if not result.reasons_histogram:
        lines.append("| _none_ | 0 |")
    if result.notes:
        lines += ["", "**Notes:**", *[f"- {note}" for note in result.notes]]
    lines += [
        "",
        "_A record may fail several rules; histogram counts rule hits, not records._",
        "",
    ]
    return "\n".join(lines)


def write_report(result: GateResult, cfg: QualityConfig, root: Path) -> tuple[Path, Path]:
    """Write JSON + Markdown reports; returns their paths."""
    report_dir = root / "reports" / "quality"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"config": cfg.model_dump(mode="json"), **result.as_dict()}
    json_path = report_dir / f"{result.dataset}.json"
    md_path = report_dir / f"{result.dataset}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(render_markdown(result, cfg))
    return json_path, md_path
