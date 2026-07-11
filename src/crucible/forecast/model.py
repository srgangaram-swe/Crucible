"""PatchTST-style probabilistic forecaster with per-window reversible normalization."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "time-series forecasting requires torch; install with "
            "`pip install crucible-data[forecast]`"
        ) from exc
    return torch


class PatchTSTConfig(BaseModel):
    """Direct multi-horizon probabilistic PatchTST model configuration."""

    context_length: int = Field(default=96, ge=16)
    horizon: int = Field(default=12, ge=1)
    patch_length: int = Field(default=16, ge=4)
    patch_stride: int = Field(default=8, ge=1)
    d_model: int = Field(default=64, ge=16)
    n_heads: int = Field(default=4, ge=1)
    n_layers: int = Field(default=3, ge=1)
    feedforward_multiplier: int = Field(default=4, ge=2)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    target_index: int = Field(default=0, ge=0)
    seed: int = 41

    @model_validator(mode="after")
    def _consistent(self) -> PatchTSTConfig:
        if self.patch_length > self.context_length:
            raise ValueError("patch_length cannot exceed context_length")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("quantiles must be unique and strictly increasing")
        if 0.5 not in self.quantiles or any(not 0 < value < 1 for value in self.quantiles):
            raise ValueError("quantiles must be in (0, 1) and include 0.5")
        return self


def build_patchtst(config: PatchTSTConfig, n_features: int) -> Any:
    """Build a seeded channel-independent patch Transformer with ordered quantiles."""
    torch = _require_torch()
    from torch import nn
    from torch.nn import functional

    if config.target_index >= n_features:
        raise ValueError("target_index is outside the feature dimension")
    torch.manual_seed(config.seed)
    n_patches = 1 + (config.context_length - config.patch_length) // config.patch_stride

    class ProbabilisticPatchTST(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_projection = nn.Linear(config.patch_length, config.d_model)
            self.position = nn.Parameter(torch.zeros(1, n_patches, config.d_model))
            self.channel_embedding = nn.Parameter(torch.zeros(1, n_features, config.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.feedforward_multiplier * config.d_model,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=config.n_layers, enable_nested_tensor=False
            )
            self.channel_score = nn.Linear(config.d_model, 1)
            self.norm = nn.LayerNorm(config.d_model)
            self.head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.horizon * len(config.quantiles)),
            )
            nn.init.trunc_normal_(self.position, std=0.02)
            nn.init.trunc_normal_(self.channel_embedding, std=0.02)

        def forward(self, inputs: Any) -> Any:
            if inputs.ndim != 3 or inputs.shape[1:] != (config.context_length, n_features):
                raise ValueError(
                    f"expected [batch, {config.context_length}, {n_features}], "
                    f"received {tuple(inputs.shape)}"
                )
            if not torch.isfinite(inputs).all():
                raise ValueError("forecast inputs must be finite")
            mean = inputs.mean(dim=1, keepdim=True)
            scale = torch.sqrt(inputs.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            normalized = (inputs - mean) / scale
            patches = normalized.transpose(1, 2).unfold(
                dimension=-1, size=config.patch_length, step=config.patch_stride
            )
            batch, channels, patch_count, _ = patches.shape
            tokens = self.patch_projection(patches).reshape(
                batch * channels, patch_count, config.d_model
            )
            encoded = self.encoder(tokens + self.position[:, :patch_count])
            channel_state = encoded.mean(dim=1).reshape(batch, channels, config.d_model)
            channel_state = channel_state + self.channel_embedding[:, :channels]
            weights = torch.softmax(self.channel_score(channel_state), dim=1)
            pooled = self.norm((channel_state * weights).sum(dim=1))
            raw = self.head(pooled).reshape(batch, config.horizon, len(config.quantiles))
            median_index = config.quantiles.index(0.5)
            if len(config.quantiles) == 3 and median_index == 1:
                median = raw[..., 1:2]
                ordered = torch.cat(
                    [
                        median - functional.softplus(raw[..., 0:1]),
                        median,
                        median + functional.softplus(raw[..., 2:3]),
                    ],
                    dim=-1,
                )
            else:
                ordered = torch.cat(
                    [
                        raw[..., :1],
                        raw[..., :1] + torch.cumsum(functional.softplus(raw[..., 1:]), dim=-1),
                    ],
                    dim=-1,
                )
            target_mean = mean[:, :, config.target_index].unsqueeze(-1)
            target_scale = scale[:, :, config.target_index].unsqueeze(-1)
            return ordered * target_scale + target_mean

    return ProbabilisticPatchTST()


def quantile_loss(predictions: Any, targets: Any, quantiles: tuple[float, ...]) -> Any:
    """Mean pinball loss across batch, forecast horizons, and quantiles."""
    torch = _require_torch()
    if predictions.ndim != 3 or targets.shape != predictions.shape[:2]:
        raise ValueError(
            "predictions must be [batch, horizon, quantile] and targets [batch, horizon]"
        )
    levels = torch.tensor(quantiles, dtype=predictions.dtype, device=predictions.device)
    errors = targets.unsqueeze(-1) - predictions
    return torch.maximum(levels * errors, (levels - 1) * errors).mean()
