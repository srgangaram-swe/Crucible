# Experiment index

Phase 8 studies use the deterministic byte-bigram proxy language model described in
[the capstone](../../RESULTS.md). Every table below is generated from three seeds and
committed as JSON, CSV, Markdown, and SVG. Artifact paths contain both the validated
config hash and the measured-result content hash.

| Study | Config | Result | Report |
|---|---|---|---|
| Dedup threshold | `adc73ba41e5f` | `77bfe300a204` | [analysis](dedup_ablation.md) |
| Domain mixture | `008655194a79` | `7355b35958c6` | [analysis](mixing_ablation.md) |
| Quality gate | `7105b682645a` | `b585e9b6f48e` | [analysis](quality_ablation.md) |
| Data scaling | `bbb62eea7f3e` | `68a3d28153ac` | [analysis](scaling_law.md) |

Reproduce all committed artifacts:

```bash
for config in configs/experiments/*.yaml; do
  crucible assay --config "$config" --out results/experiments
done
```
