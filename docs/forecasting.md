# Probabilistic time-series forecasting

Crucible 0.10 adds a real supervised learning path for numeric time series. It is
separate from the byte-language-model reference trainer because timestamped numeric
features, chronological validation, and forecast metrics are different contracts from
token shards.

## Architecture

The model is PatchTST-inspired: each past-observed channel is normalized using only
its context window (RevIN), split into overlapping patches, projected through a shared
norm-first Transformer encoder, and pooled across channels with learned attention. A
direct head predicts every horizon and the 0.1/0.5/0.9 quantiles in one pass. Softplus
offsets around the median make quantile ordering structural rather than a penalty.

The implementation follows the core ideas in
[PatchTST](https://openreview.net/forum?id=Jbdc0vTOcol) and
[Reversible Instance Normalization](https://openreview.net/forum?id=cGDAkQo1C0p),
while adding cross-channel pooling for past-observed covariates. It is not a claim of
paper-level reproduction.

```text
[batch, context, channel]
  -> context-only RevIN
  -> overlapping channel-wise patches
  -> shared Transformer encoder
  -> learned channel pooling
  -> direct [horizon, quantile] head
  -> validation-fitted quantile calibration
```

The training loop is ordinary PyTorch, not a proxy: AdamW, warmup plus cosine decay,
pinball loss, gradient clipping, deterministic DataLoader shuffling, validation-only
early stopping, atomic best checkpoints, and best-weight restoration before the test
split is opened. Checkpoints include model/optimizer/scheduler state, RNG state,
validated configs, data hash, and code version.

## Leakage controls

- timestamps must be finite, unique, and strictly increasing;
- preprocessing statistics are fit through `train_end` only;
- each target interval lies wholly inside one split;
- validation and test are separated by an embargo at least as long as the horizon;
- validation selects the checkpoint and calibrates quantiles; test is evaluated once;
- future test sentinels cannot change training/validation tensors or scaler state.

## Run the reference benchmark

```bash
pip install -e '.[forecast]'
crucible forecast --config configs/forecast_financial.yaml --out results/forecast
```

The committed run `19c5e9fc804f6f61` uses 1,800 synthetic daily observations,
context 96, horizon 5, 157,328 trainable parameters, and early-stops at epoch 14
(best epoch 4). On its sealed rolling-origin test windows:

| model | MAE | RMSE | MASE | directional accuracy |
|---|---:|---:|---:|---:|
| Patch forecaster | 0.01235 | 0.01708 | 0.836 | 0.570 |
| persistence | 0.01702 | 0.02342 | 1.153 | 0.520 |
| seasonal naive | 0.01709 | 0.02388 | 1.158 | 0.562 |
| zero return | **0.01224** | 0.01724 | **0.829** | 0.000 |

The model improves MAE by 27.5% over persistence (moving-block bootstrap 95% CI for
skill: [22.4%, 31.5%]) and 27.7% over seasonal naive. It
does **not** beat zero-return point MAE (0.9% worse). Mean pinball loss is 0.00424,
the nominal 80% interval covers 75.5%, and quantile crossing is exactly zero. These
are deterministic synthetic measurements proving the pipeline and optimization work;
they are not evidence of financial alpha.

Evaluation artifacts include [metrics](../results/forecast/19c5e9fc804f6f61/metrics.json),
[predictions](../results/forecast/19c5e9fc804f6f61/predictions.csv),
[forecast intervals](../results/forecast/19c5e9fc804f6f61/forecast.svg),
[training curves](../results/forecast/19c5e9fc804f6f61/training.svg),
[residuals](../results/forecast/19c5e9fc804f6f61/residuals.svg),
[calibration](../results/forecast/19c5e9fc804f6f61/calibration.svg), and
[per-horizon errors](../results/forecast/19c5e9fc804f6f61/metrics_by_horizon.svg).

## Bring real financial data later

Set `input_csv`, `timestamp_column`, `target_column`, and `feature_columns` in a copy
of the YAML config. Inputs are vendor-neutral numeric columns, so adjusted returns,
volume, realized volatility, and point-in-time features can be supplied without a new
model API.

Before interpreting real results, add an exchange calendar, point-in-time universe
membership, corporate-action policy, availability timestamps for fundamentals/news,
transaction costs, walk-forward re-training, and a multiple-testing protocol. Backtests
without those controls are not decision-grade.
