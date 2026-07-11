# Deduplication threshold ablation

We compare MinHash/Jaccard thresholds 0.40, 0.50, and 0.60 at exactly 20,000
proxy-model training bytes per arm. Across seeds 11, 23, and 37, mean held-out
cross-entropy was 2.690, 2.686, and 2.697 respectively. All bootstrap intervals
overlap substantially; this experiment does **not** establish a downstream winner.

The negative result is plausible at this corpus and compute scale: fixed compute cycles
the retained content, while the threshold primarily changes which redundant examples
are available. It should not be generalized to web-scale training. Lee et al.,
[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499),
motivate the hypothesis at realistic scale; our result only validates the experimental
path and reports its small-scale outcome honestly.

Source: `configs/experiments/dedup_ablation.yaml`; config `adc73ba41e5f`;
result `77bfe300a204`.
