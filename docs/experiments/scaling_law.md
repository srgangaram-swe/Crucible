# Data-scaling fit

After threshold-0.5 deduplication, proxy training uses 20k, 50k, 100k, and 200k
total bytes. Mean held-out loss falls from 2.686 to 2.341. The fitted power-law
exponent for total tokens is -0.0588 (95% bootstrap interval [-0.0688, -0.0538]);
against unique post-dedup content it is -0.0832 [-0.0943, -0.0748]. At 200k total
bytes only about 89k are unique, making the onset of repeated-content saturation
explicit.

These are four points from three small synthetic seeds and a bigram proxy—not a
frontier-model scaling law. The exponents are descriptive within the measured range;
extrapolation would be statistically and scientifically unjustified.

Source: `configs/experiments/scaling_law.yaml`; config `ddf9ad49292e`;
result `3a75562b1955`.
