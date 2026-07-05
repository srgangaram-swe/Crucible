# Security Policy

Crucible is a local-first research tool; it ships no network service by
default. The optional FastAPI/Streamlit components (Phase 7) are meant for
local or trusted-network use and have no authentication story — do not expose
them to the public internet.

## Reporting a vulnerability

Open a GitHub security advisory or email saifryangangaram@gmail.com. Please
include a minimal reproduction. You should hear back within a week.

## Scope notes

- The synthetic corpus generator plants *synthetic* PII (`user123@example.com`,
  555-prefixed phone numbers) so quality gates can be tested; no real personal
  data is used or included anywhere in the repo.
- SQL entered via `crucible sql` runs with DuckDB's default local privileges
  against your own catalog; treat it like any local SQL shell.
