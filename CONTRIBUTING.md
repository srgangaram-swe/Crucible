# Contributing

Crucible is a personal research-infrastructure project, but issues and PRs are
welcome. The bar is: everything runs offline on a laptop CPU, everything is
typed, tested, and reproducible from a config.

## Setup

```bash
make install   # venv + package + dev tools
make hooks     # pre-commit
make gate      # lint + types + tests + coverage + smoke — must pass before a PR
```

## Ground rules

- **Offline-first.** The core package must work with no extras, no Docker, no
  network, no GPU. Optional capabilities go behind extras (`.[stream]`,
  `.[hf]`, `.[quality]`, `.[dedup]`, `.[train]`, `.[serve]`) with an offline
  fallback and identical tests where feasible.
- **Determinism.** Any code that touches randomness takes a seed and derives
  per-concern RNG streams (see `crucible.synth`). No wall-clock reads in
  anything that defines dataset content.
- **Ground truth is read-only for pipelines.** `gt_*` columns exist so the
  research harness can score pipeline stages. Pipeline code must never read
  them; that separation is what makes reported precision/recall honest.
- **Measured, not claimed.** Numbers in docs come from re-executable runs, or
  they are labeled illustrative.
- **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `ci:`,
  `refactor:`, `perf:`, `chore:`), small and reviewable. One feature branch
  per phase/topic; CI must be green to merge.

## Tests

`pytest` for unit + integration; new modules need both happy-path and failure
tests. Coverage is enforced (see `fail_under` in pyproject). Anything optional
(Kafka, HF, GPU) is `skipif`-gated so the default suite stays hermetic.
