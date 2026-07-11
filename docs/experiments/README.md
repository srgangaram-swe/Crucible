# Experiment index

Phase 8 studies use the deterministic byte-bigram proxy language model described in
[the capstone](../../RESULTS.md). Every table below is generated from three seeds and
committed as JSON, CSV, Markdown, and SVG. Artifact paths contain both the validated
config hash and the measured-result content hash.

| Study | Config | Result | Report |
|---|---|---|---|
| Dedup threshold | `437c528d0822` | `988aad638f56` | [analysis](dedup_ablation.md) |
| Domain mixture | `eea2fcac3c38` | `ae32b89a0f61` | [analysis](mixing_ablation.md) |
| Quality gate | `46c7fe66f44b` | `6e32ce801a86` | [analysis](quality_ablation.md) |
| Data scaling | `ddf9ad49292e` | `3a75562b1955` | [analysis](scaling_law.md) |

Reproduce all committed artifacts:

```bash
for config in configs/experiments/*.yaml; do
  crucible assay --config "$config" --out results/experiments
done
```
