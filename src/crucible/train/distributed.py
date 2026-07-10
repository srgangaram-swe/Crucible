"""DDP / FSDP entrypoints for the reference trainer.

Launch with torchrun against a catalog of prebuilt shards, e.g.:

    torchrun --nproc_per_node 2 -m crucible.train.distributed \
        --root data/crucible --shards synth_shards --steps 20 --mode ddp

Data sharding is rank-round-robin over the (deterministic) shard iterator:
every rank streams the same seeded order and keeps sequences where
``index % world_size == rank`` — no coordination, no overlap, and the
union of ranks sees exactly the single-process stream. Gradients are then
averaged by DDP/FSDP, so a fixed global batch produces the same optimizer
trajectory as single-process training; the loss-parity test asserts that
(gloo backend, CPU, opt-in via CRUCIBLE_RUN_DDP=1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crucible.shards import ShardReader
from crucible.storage import Catalog
from crucible.train.loop import TrainConfig
from crucible.train.model import _require_torch, build_model


def rank_batches(reader: ShardReader, cfg: TrainConfig, rank: int, world_size: int) -> Any:
    """Round-robin split of the deterministic stream across ranks."""
    torch = _require_torch()
    epoch = 0
    per_rank = max(1, cfg.batch_size // world_size)
    iterator = reader.iterate(epoch=epoch)
    index = 0
    rows: list[list[int]] = []
    while True:
        try:
            sequence = next(iterator)
        except StopIteration:
            epoch += 1
            iterator = reader.iterate(epoch=epoch)
            continue
        if index % world_size == rank:
            rows.append(sequence)
            if len(rows) == per_rank:
                yield torch.tensor(rows, dtype=torch.long)
                rows = []
        index += 1


def distributed_train(
    catalog: Catalog, cfg: TrainConfig, mode: str, rank: int, world_size: int
) -> list[float]:
    """Train under an initialized process group; returns per-step losses."""
    torch = _require_torch()
    from torch.nn import functional
    from torch.nn.parallel import DistributedDataParallel

    torch.manual_seed(cfg.seed)
    model = build_model(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        seed=cfg.seed,
    )
    wrapped: Any
    if mode == "ddp":
        wrapped = DistributedDataParallel(model)
    elif mode == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel

        wrapped = FullyShardedDataParallel(model)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected ddp or fsdp")

    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=cfg.lr)
    reader = ShardReader(
        catalog, cfg.shards_dataset, seed=cfg.seed, shuffle_buffer=cfg.shuffle_buffer
    )
    batches = rank_batches(reader, cfg, rank, world_size)

    losses: list[float] = []
    for _ in range(cfg.steps):
        batch = next(batches)
        x, y = batch[:, :-1], batch[:, 1:]
        logits = wrapped(x)
        loss = functional.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


def main() -> None:  # pragma: no cover - torchrun entrypoint
    torch = _require_torch()
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo" if not torch.cuda.is_available() else "nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    cfg = TrainConfig(
        shards_dataset=args.shards,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    losses = distributed_train(Catalog(args.root), cfg, args.mode, rank, world_size)
    if rank == 0:
        print(json.dumps({"mode": args.mode, "world_size": world_size, "losses": losses}))
    dist.destroy_process_group()


if __name__ == "__main__":  # pragma: no cover
    main()
