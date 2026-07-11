# Quality-gate ablation

The arms are ungated, default rules, and default plus repeated-sentence rejection.
Keep rates were 1.000, 0.900, and 0.868. Mean held-out losses were 2.6830, 2.6815,
and 2.6821; their bootstrap intervals overlap. The stricter rule discards another
3.2% of documents without measurable benefit in this proxy study.

This agrees with the precision/keep-rate warning in [quality.md](../quality.md): the
default gate exactly catches planted junk, while repeated-sentence rejection is a
policy tradeoff rather than a universally beneficial heuristic.

Source: `configs/experiments/quality_ablation.yaml`; config `7105b682645a`;
result `b585e9b6f48e`.
