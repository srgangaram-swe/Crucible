"""Deterministic optimization, early stopping, checkpointing, and inference."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from crucible import __version__
from crucible.forecast.data import WindowSet
from crucible.forecast.model import PatchTSTConfig, _require_torch, build_patchtst, quantile_loss


class ForecastTrainConfig(BaseModel):
    epochs: int = Field(default=60, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    warmup_epochs: int = Field(default=3, ge=0)
    patience: int = Field(default=10, ge=1)
    min_delta: float = Field(default=1e-5, ge=0)
    gradient_clip_norm: float = Field(default=1.0, gt=0)
    seed: int = 41
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class ForecastTrainResult:
    best_epoch: int
    epochs_completed: int
    best_validation_loss: float
    trainable_parameters: int
    elapsed_seconds: float
    device: str
    stop_reason: str
    history: list[dict[str, float]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _seed_everything(seed: int) -> None:
    torch = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _loader(windows: WindowSet, batch_size: int, shuffle: bool, seed: int) -> Any:
    torch = _require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.from_numpy(windows.inputs), torch.from_numpy(windows.targets))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def _device(config: ForecastTrainConfig) -> str:
    torch = _require_torch()
    if config.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return config.device


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    model_config: PatchTSTConfig,
    train_config: ForecastTrainConfig,
    n_features: int,
    epoch: int,
    validation_loss: float,
    dataset_hash: str,
) -> None:
    torch = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "model_config": model_config.model_dump(mode="json"),
            "train_config": train_config.model_dump(mode="json"),
            "n_features": n_features,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "dataset_hash": dataset_hash,
            "code_version": __version__,
        },
        temporary,
    )
    temporary.replace(path)


def train_forecaster(
    model_config: PatchTSTConfig,
    train_config: ForecastTrainConfig,
    train: WindowSet,
    validation: WindowSet,
    checkpoint_path: Path,
) -> tuple[Any, ForecastTrainResult]:
    """Fit on train only, select on validation only, and restore the best weights."""
    torch = _require_torch()
    _seed_everything(train_config.seed)
    device = _device(train_config)
    n_features = train.inputs.shape[2]
    dataset_hash = hashlib.sha256(
        train.inputs.tobytes()
        + train.targets.tobytes()
        + validation.inputs.tobytes()
        + validation.targets.tobytes()
    ).hexdigest()
    model = build_patchtst(model_config, n_features).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    def learning_rate_multiplier(epoch: int) -> float:
        if train_config.warmup_epochs and epoch < train_config.warmup_epochs:
            return (epoch + 1) / train_config.warmup_epochs
        remaining = max(1, train_config.epochs - train_config.warmup_epochs)
        progress = (epoch - train_config.warmup_epochs) / remaining
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    train_loader = _loader(train, train_config.batch_size, True, train_config.seed)
    validation_loader = _loader(validation, train_config.batch_size, False, train_config.seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(train_config.epochs):
        model.train()
        train_total = 0.0
        train_examples = 0
        gradient_norm_total = 0.0
        gradient_steps = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = quantile_loss(predictions, targets, model_config.quantiles)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.gradient_clip_norm
            )
            optimizer.step()
            train_total += float(loss.item()) * len(inputs)
            train_examples += len(inputs)
            gradient_norm_total += float(gradient_norm)
            gradient_steps += 1
        model.eval()
        validation_total = 0.0
        validation_examples = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                loss = quantile_loss(model(inputs), targets, model_config.quantiles)
                validation_total += float(loss.item()) * len(inputs)
                validation_examples += len(inputs)
        train_loss = train_total / train_examples
        validation_loss = validation_total / validation_examples
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "gradient_norm": gradient_norm_total / gradient_steps,
            }
        )
        if validation_loss < best_loss - train_config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            _save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                model_config=model_config,
                train_config=train_config,
                n_features=n_features,
                epoch=best_epoch,
                validation_loss=best_loss,
                dataset_hash=dataset_hash,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= train_config.patience:
            break
    if best_state is None:
        raise RuntimeError("training completed without a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    result = ForecastTrainResult(
        best_epoch=best_epoch,
        epochs_completed=len(history),
        best_validation_loss=best_loss,
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        elapsed_seconds=time.perf_counter() - started,
        device=device,
        stop_reason=("early_stopping" if stale_epochs >= train_config.patience else "max_epochs"),
        history=history,
    )
    return model, result


def predict(model: Any, windows: WindowSet, batch_size: int, device: str) -> np.ndarray:
    torch = _require_torch()
    loader = _loader(windows, batch_size, False, 0)
    batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, _targets in loader:
            batches.append(model(inputs.to(device)).cpu().numpy())
    return np.concatenate(batches).astype(np.float32)


def load_forecaster(checkpoint_path: Path, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    torch = _require_torch()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = PatchTSTConfig.model_validate(payload["model_config"])
    model = build_patchtst(config, int(payload["n_features"]))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    metadata = {
        "epoch": int(payload["epoch"]),
        "validation_loss": float(payload["validation_loss"]),
        "model_config": payload["model_config"],
        "train_config": payload["train_config"],
        "n_features": int(payload["n_features"]),
        "dataset_hash": str(payload["dataset_hash"]),
        "code_version": str(payload["code_version"]),
    }
    return model, metadata


def write_history(path: Path, result: ForecastTrainResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
