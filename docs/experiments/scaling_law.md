# Data-scaling fit

After threshold-0.5 deduplication, proxy training uses 20k, 50k, 100k, and 200k
total bytes. Mean held-out loss falls from 2.686 to 2.341. The fitted power-law
exponent for total tokens is -0.0599 (95% bootstrap interval [-0.0709, -0.0539]);
against unique post-dedup content it is -0.0760 [-0.0885, -0.0685]. At 200k total
bytes only about 109k are unique, making the onset of repeated-content saturation
explicit.

These are four points from three small synthetic seeds and a bigram proxy—not a
frontier-model scaling law. The exponents are descriptive within the measured range;
extrapolation would be statistically and scientifically unjustified.

Source: `configs/experiments/scaling_law.yaml`; config `bbb62eea7f3e`;
result `68a3d28153ac`.
