# Crucible Phase 8 results

## Method

Four seed-controlled studies exercise the same data decisions as the main pipeline.
The evaluator is a Laplace-smoothed byte-bigram language model: intentionally small,
offline, deterministic, and sensitive to corpus composition. Each arm is trained on
an identical byte budget where comparisons require equal compute. Evaluation uses
held-out clean synthetic documents; mixture selection uses a separate proxy split.

Every metric row records its seed and arm. Reports aggregate three seeds with
deterministic percentile-bootstrap 95% confidence intervals. The config SHA-256
includes the validated YAML and Crucible version; the result SHA-256 includes every
measured row. Thus every number below resolves to a committed config and artifact.

## Findings

- Dedup threshold: losses 2.689/2.686/2.697 for thresholds 0.4/0.5/0.6. No winner;
  uncertainty overlaps.
- Mixture: code-heavy is worse (2.804) than observed (2.652) or uniform (2.658).
  Proxy guidance selected an existing grid arm and offered no consistent gain.
- Quality: default gating removes 12.3% and changes loss from 2.6830 to 2.6815;
  repeated-sentence filtering removes another 2.6% with no measurable benefit.
- Scaling: loss falls 2.686→2.341 from 20k→200k total bytes. Fitted exponents are
  -0.0588 by total tokens and -0.0832 by unique post-dedup content, with saturation
  visible once the corpus is replayed.

These findings are engineering validation and small-scale evidence, not claims about
large language models. Synthetic templates, three seeds, and a bigram proxy sharply
limit external validity. Negative results are retained because suppressing them would
make this a demo rather than a research harness.

See [the experiment index](docs/experiments/README.md) for configs, result hashes,
per-study interpretation, plots, and exact reproduction commands.

## Probabilistic forecasting benchmark

The v0.10 PatchTST-style forecaster is trained on 1,800 deterministic synthetic
regime-switching daily returns with a sealed chronological split and five-step embargo.
Held-out MAE is 0.01235: 27.5% better than persistence (moving-block bootstrap 95%
skill interval [22.4%, 31.5%]) and 27.7% better than seasonal
naive, but 0.9% worse than a zero-return point forecast. Directional accuracy is 57.0%,
mean pinball loss is 0.00424, nominal 80% interval coverage is 75.5%, and quantile
crossing is zero. See [the model report](docs/forecasting.md) and committed evaluation
plots. This is implementation evidence on a controlled process, not financial alpha.
