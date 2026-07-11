# Orchestration and metadata control plane

Phase 7 adds an intentionally small local control plane without turning the data
pipeline into a framework. `DagRunner` accepts typed tasks, validates missing
dependencies and cycles before execution, uses a stable topological order, retries
only up to each task's declared bound, and atomically persists every run under
`<root>/runs/<run-id>/run.json`.

The built-in `orchestrate` command encodes the production invariant that promotion
must complete before deduplication. Its idempotency key is the SHA-256 of the bronze
manifest and both validated configs. A completed fingerprint is skipped; operators
must pass `--force` to rerun it.

```bash
crucible orchestrate --dataset synth --root data/crucible
crucible runs --root data/crucible
crucible metrics --root data/crucible
```

Every attempt emits append-only JSONL metrics containing status, duration, input and
output rows, derived row throughput, attempt number, and a bounded error string. A
failed run is persisted before the exception reaches the CLI.

The optional serving extra provides a read-only metadata API:

```bash
pip install -e '.[serve]'
crucible serve --root data/crucible --host 127.0.0.1 --port 8000
streamlit run src/crucible/dashboard.py
```

Endpoints live under `/v1`: `catalog`, `versions/{dataset}`, `lineage`,
`reports/{quality|dedup}/{dataset}`, `runs`, and `metrics`. `/healthz` is reserved
for process health. The application factory takes a catalog root, so tests and
deployments never depend on process-global paths. The API is deliberately read-only;
pipeline mutations remain explicit CLI operations.
