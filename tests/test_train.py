"""Reference-trainer tests (skip-gated on torch; CI runs them in the train job)."""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from crucible.ingest import JsonlSource, land  # noqa: E402
from crucible.quality import QualityConfig, run_gate  # noqa: E402
from crucible.shards import ShardConfig, build_shards  # noqa: E402
from crucible.storage import Catalog  # noqa: E402
from crucible.synth import SynthConfig, generate_corpus, write_jsonl  # noqa: E402
from crucible.train import TrainConfig, train  # noqa: E402

CFG = TrainConfig(shards_dataset="synth_shards", steps=20, batch_size=8, seed=0)


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Catalog:
    base = tmp_path_factory.mktemp("train")
    corpus = base / "corpus.jsonl"
    write_jsonl(generate_corpus(SynthConfig(seed=13, n_docs=150)), corpus)
    cat = Catalog(base / "catalog")
    land(JsonlSource(corpus, batch_size=60), cat, "synth", "test")
    run_gate(cat, "synth", QualityConfig())
    build_shards(cat, "synth", ShardConfig(seq_len=64, sequences_per_shard=50))
    return cat


def test_loss_decreases_on_cpu(catalog: Catalog) -> None:
    result = train(catalog, CFG)
    assert result.device in ("cpu", "cuda")
    assert result.final_loss < result.initial_loss
    assert result.mean_loss_last5 < result.initial_loss * 0.8  # measured: ~5.6 -> <2
    assert result.tokens_per_second > 0


def test_training_is_deterministic(catalog: Catalog) -> None:
    first = train(catalog, CFG)
    second = train(catalog, CFG)
    assert first.final_loss == second.final_loss
    different_seed = train(catalog, CFG.model_copy(update={"seed": 1}))
    assert different_seed.final_loss != first.final_loss


def test_checkpoint_resume_matches_uninterrupted_run(catalog: Catalog, tmp_path: Path) -> None:
    """train(20) == train(10) + resume(10): only possible because the shard
    iterator's replay-resume is exact."""
    full = train(catalog, CFG)

    ckpt = tmp_path / "step10.pt"
    train(catalog, CFG.model_copy(update={"steps": 10}), checkpoint_path=ckpt)
    resumed = train(catalog, CFG.model_copy(update={"steps": 10}), resume_from=ckpt)
    assert resumed.final_loss == pytest.approx(full.final_loss, abs=1e-5)
    assert resumed.mean_loss_last5 == pytest.approx(full.mean_loss_last5, abs=1e-5)


def _ddp_worker(rank: int, world_size: int, root: str, out: str) -> None:
    """Module-level so mp.spawn can pickle it."""
    import json

    import torch.distributed as dist

    from crucible.train.distributed import distributed_train

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    losses = distributed_train(Catalog(Path(root)), CFG, "ddp", rank, world_size)
    if rank == 0:
        Path(out).write_text(json.dumps(losses))
    dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("CRUCIBLE_RUN_DDP") != "1",
    reason="set CRUCIBLE_RUN_DDP=1 to run the 2-process gloo parity test",
)
def test_ddp_two_process_loss_parity(catalog: Catalog, tmp_path: Path) -> None:
    """Fixed global batch + rank-round-robin sharding -> DDP(2) matches
    single-process losses closely, step for step."""
    import torch.multiprocessing as mp

    out = tmp_path / "ddp_losses.json"
    mp.spawn(_ddp_worker, args=(2, str(catalog.root), str(out)), nprocs=2, join=True)

    import json

    ddp_losses = json.loads(out.read_text())
    single = train(catalog, CFG)
    assert len(ddp_losses) == CFG.steps
    assert ddp_losses[-1] == pytest.approx(single.final_loss, abs=0.15)
