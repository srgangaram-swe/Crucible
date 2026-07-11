# Quality-gate ablation

The arms are ungated, default rules, and default plus repeated-sentence rejection.
Keep rates were 1.000, 0.877, and 0.851. Mean held-out losses were 2.6830, 2.6815,
and 2.6821; their bootstrap intervals overlap. The stricter rule discards another
2.6% of training documents without measurable benefit in this proxy study.

This agrees with the precision/keep-rate warning in [quality.md](../quality.md): the
default gate exactly catches planted junk, while repeated-sentence rejection is a
policy tradeoff rather than a universally beneficial heuristic.

Source: `configs/experiments/quality_ablation.yaml`; config `46c7fe66f44b`;
result `6e32ce801a86`.
