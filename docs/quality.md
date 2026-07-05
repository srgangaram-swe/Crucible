# Quality gates

The quality subsystem promotes bronze to silver through validating rules,
parks failures in quarantine, and detects distribution drift. Everything here
runs offline with core dependencies only; pandera (`.[quality]`) adds an
optional declarative second opinion.

## Rule engine

Rules are pure predicates over a record's text, inspired by the web-corpus
filtering literature — C4 (Raffel et al., 2020, arXiv:1910.10683) and
RefinedWeb (Penedo et al., 2023, arXiv:2306.01116) — but scaled down to
heuristics whose hit rates are verified *exactly* against the synthetic
corpus's planted defects:

| Rule | Catches (planted defect) | Mechanism |
|---|---|---|
| `non_empty` | `junk_empty` | stripped length > 0 |
| `min_words` | `junk_short` | word count ≥ `min_words` (default 5) |
| `no_mojibake` | `junk_mojibake` | double-encoding digraphs, U+FFFD, control chars |
| `no_boilerplate_markers` | `junk_boilerplate` | ≥ `boilerplate_marker_threshold` web-chrome phrases |
| `no_pii` | `pii` | email/phone/SSN regexes |
| `no_repeated_sentences` | (opt-in) | duplicated-sentence fraction > ratio |

## Measured results

From the seed-42, 400-record smoke corpus (re-run with `make smoke`; the
numbers below are asserted by the smoke test itself):

- **Default rule set: precision 1.0, recall 1.0** — all 24 junk and 16 PII
  records quarantined, zero clean/duplicate records lost. Per-kind recall is
  1.0 for every planted defect kind.
- **Adding `no_repeated_sentences` (naive, any duplicate)**: precision drops
  to **0.23** — it quarantines ~30% of the corpus, mostly clean forum-Q&A
  docs whose template legitimately reuses answer phrasing.
- **Adding it ratio-based (default 0.3)**: precision **0.87**, recall still
  1.0 — six clean false positives for zero recall gain, which is why the rule
  ships opt-in. Quantifying exactly this aggressiveness/keep-rate tradeoff is
  the Phase 8 quality ablation.

Duplicates (exact/near) are deliberately *not* quality defects: a gate that
quarantined them would be over-reaching into dedup's job, and is scored down
via precision accordingly (see `crucible.assay.scoring`).

## Promotion semantics

`crucible promote --dataset D` reads `bronze/D`, evaluates every enabled rule
per record, and:

- writes survivors to `silver/D` (content-addressed parts, deterministic
  across re-runs),
- writes failures to `quarantine/D` with a `reject_reasons` column,
- writes JSON + Markdown reports to `<root>/reports/quality/`,
- **blocks everything** (exit 1, empty silver) if the reject rate exceeds
  `max_reject_rate`.

## Drift detection

`crucible drift --dataset REF --against CUR` computes the Population
Stability Index over the source mix and the binned document-length
distribution, with the standard thresholds (<0.1 none, 0.1–0.25 moderate,
>0.25 major; Siddiqi 2006). Measured on the smoke corpus: a mixture skewed to
85% code registers source PSI ≈ 2.48 (major); an identically-distributed
sample registers none.

Profiles are JSON artifacts (`DatasetProfile.to_json`), so a reference profile
can be pinned alongside a dataset version and compared against any later
ingest.
