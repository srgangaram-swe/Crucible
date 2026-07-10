"""Reference trainer: a deliberately small transformer over crucible shards.

The model is tiny on purpose — the deliverable is the DATA path (shard
streaming, exact resume, deterministic order) and the training
infrastructure (CPU / DDP / FSDP entrypoints), not model quality. torch
lives behind the ``train`` extra; importing this package without it raises
a helpful error, and everything else in crucible works untouched.
"""

from crucible.train.loop import TrainConfig, TrainResult, train

__all__ = ["TrainConfig", "TrainResult", "train"]
