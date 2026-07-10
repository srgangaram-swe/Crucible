"""A minimal decoder-only transformer (byte vocabulary).

Small enough to take a few optimizer steps per second on a laptop CPU,
structured conventionally (embeddings -> causal self-attention blocks ->
tied-ish LM head) so the DDP/FSDP wrappers exercise realistic module trees.
"""

from __future__ import annotations

from typing import Any


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "the reference trainer requires torch; "
            "install with: pip install 'crucible-data[train]'"
        ) from exc
    return torch


def build_model(
    vocab_size: int = 259,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    max_seq_len: int = 512,
    seed: int = 0,
) -> Any:
    """Construct the reference model deterministically (seeded init)."""
    torch = _require_torch()
    from torch import nn

    torch.manual_seed(seed)

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                batch_first=True,
                dropout=0.0,  # determinism over regularization, on purpose
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers, enable_nested_tensor=False
            )
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, tokens: Any) -> Any:
            seq_len = tokens.shape[1]
            positions = torch.arange(seq_len, device=tokens.device)
            hidden = self.token_embedding(tokens) + self.position_embedding(positions)
            causal = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tokens.device)
            hidden = self.blocks(hidden, mask=causal, is_causal=True)
            return self.lm_head(self.norm(hidden))

    return TinyTransformer()
