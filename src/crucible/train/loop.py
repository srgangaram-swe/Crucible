"""Training loop over crucible shards: deterministic, checkpointable, measured.

The loop consumes :class:`crucible.shards.ShardReader` directly — batches
are lists of ``seq_len + 1`` token sequences, split into ``x = t[:, :-1]``
and ``y = t[:, 1:]``. A checkpoint is (model state, optimizer state,
iterator state, step); resuming from it continues the *exact* run: the
resume-equivalence test asserts train(20) == train(10) + resume(10) to the
last logit, which is only possible because the shard iterator's replay
resume is exact.

Throughput numbers in :class:`TrainResult` are measured wall-clock, and are
the seed data for the Phase 6 benchmark runner.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from crucible.shards import ShardReader
from crucible.storage import Catalog
from crucible.train.model import _require_torch, build_model


class TrainConfig(BaseModel):
    """One training run over a gold shards dataset."""

    shards_dataset: str
    steps: int = Field(default=20, ge=1)
    batch_size: int = Field(default=8, ge=1)
    lr: float = Field(default=3e-3, gt=0)
    d_model: int = Field(default=64, ge=8)
    n_heads: int = Field(default=4, ge=1)
    n_layers: int = Field(default=2, ge=1)
    seed: int = 0
    shuffle_buffer: int = Field(default=256, ge=1)
    vocab_size: int = Field(default=259, ge=2)


@dataclass(frozen=True, slots=True)
class TrainResult:
    steps: int
    initial_loss: float
    final_loss: float
    mean_loss_last5: float
    tokens_seen: int
    tokens_per_second: float
    elapsed_s: float
    device: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _batches(reader: ShardReader, cfg: TrainConfig, resume_state: dict[str, int] | None) -> Any:
    """Yield (batch_tensor, iterator_state) pairs, cycling epochs as needed."""
    torch = _require_torch()
    epoch = resume_state["epoch"] if resume_state else 0
    iterator = reader.iterate(epoch=epoch, resume_state=resume_state)
    while True:
        rows: list[list[int]] = []
        while len(rows) < cfg.batch_size:
            try:
                rows.append(next(iterator))
            except StopIteration:
                epoch += 1
                iterator = reader.iterate(epoch=epoch)
                if not rows:
                    continue
                break
        yield torch.tensor(rows, dtype=torch.long), iterator.state()


def train(
    catalog: Catalog,
    cfg: TrainConfig,
    checkpoint_path: Path | None = None,
    resume_from: Path | None = None,
) -> TrainResult:
    """Train ``cfg.steps`` optimizer steps on CPU (or CUDA if available)."""
    torch = _require_torch()
    from torch.nn import functional

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        seed=cfg.seed,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    iterator_state: dict[str, int] | None = None
    start_step = 0
    if resume_from is not None:
        payload = torch.load(resume_from, weights_only=True)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        iterator_state = json.loads(payload["iterator_state"])
        start_step = int(payload["step"])

    reader = ShardReader(
        catalog, cfg.shards_dataset, seed=cfg.seed, shuffle_buffer=cfg.shuffle_buffer
    )
    batch_stream = _batches(reader, cfg, iterator_state)

    started = time.perf_counter()
    losses: list[float] = []
    tokens_seen = 0
    last_iterator_state: dict[str, int] = iterator_state or {"epoch": 0, "consumed": 0}
    for _step in range(start_step, start_step + cfg.steps):
        batch, last_iterator_state = next(batch_stream)
        batch = batch.to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = functional.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        tokens_seen += int(x.numel())

    elapsed = time.perf_counter() - started
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iterator_state": json.dumps(last_iterator_state),
                "step": start_step + cfg.steps,
                "config": cfg.model_dump(mode="json"),
            },
            checkpoint_path,
        )

    return TrainResult(
        steps=cfg.steps,
        initial_loss=round(losses[0], 4),
        final_loss=round(losses[-1], 4),
        mean_loss_last5=round(sum(losses[-5:]) / len(losses[-5:]), 4),
        tokens_seen=tokens_seen,
        tokens_per_second=round(tokens_seen / elapsed, 1),
        elapsed_s=round(elapsed, 3),
        device=device,
    )
