"""Training shards: tokenize curated silver into streaming, resumable shards.

Shards are the gold layer: fixed-length packed token sequences stored as
content-addressed Parquet parts, so every contract the platform already has
— manifests, snapshots, lineage, DuckDB views, byte-identical rebuilds —
applies to training data too. (WebDataset tars / MosaicML MDS solve the
same problem; Parquet is chosen here so one format serves the whole
refinery, and the reader below provides the streaming semantics.)

Determinism: same (silver content, ShardConfig) -> byte-identical shards.
Document order is a seeded permutation over id-sorted rows (never row
position — part read order is arbitrary); packing concatenates
``tokens + [EOS]`` and slices into ``seq_len + 1``-token sequences (the
trainer uses ``x = t[:-1], y = t[1:]``), dropping the trailing partial.

The tokenizer is byte-level (vocab 256 + PAD/BOS/EOS): zero dependencies,
no network, no trained vocabulary to version — the right fidelity for a
reference data path. Swapping in a real subword tokenizer is a config
change the experiment harness can study later.

Reading: :class:`ShardReader` streams sequences with a seeded shuffle
buffer and *exact* resume — its checkpoint is ``(epoch, consumed)`` and
resuming deterministically replays and discards the first ``consumed``
sequences (O(consumed), honest and simple at this scale; block-skipping is
a documented scale-up).
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, Field

from crucible.lineage import dataset_ref, emit_event
from crucible.storage import Catalog, Layer
from crucible.versioning import build_manifest, snapshot_stage


class ByteTokenizer:
    """Byte-level tokenizer: 256 byte values + PAD/BOS/EOS specials."""

    PAD = 256
    BOS = 257
    EOS = 258
    vocab_size = 259

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(token for token in tokens if token < 256).decode("utf-8", errors="replace")


class ShardConfig(BaseModel):
    """Shard-building policy; defaults sized for the synthetic corpus."""

    text_column: str = "text"
    id_column: str = "id"
    seq_len: int = Field(default=256, ge=8)  # stored sequences carry seq_len + 1 tokens
    sequences_per_shard: int = Field(default=512, ge=1)
    seed: int = 0


@dataclass(frozen=True, slots=True)
class ShardResult:
    dataset: str
    shards_dataset: str
    n_docs: int
    n_sequences: int
    n_tokens: int
    dropped_tail_tokens: int
    shard_parts: int
    vocab_size: int
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_shards(
    catalog: Catalog, dataset: str, cfg: ShardConfig, shards_dataset: str | None = None
) -> ShardResult:
    """Tokenize + pack silver/<dataset> into gold/<shards_dataset>."""
    started = time.perf_counter()
    shards_dataset = shards_dataset or f"{dataset}_shards"
    silver_manifest = build_manifest(catalog, Layer.SILVER, dataset)
    table = catalog.read(Layer.SILVER, dataset)

    ids = [str(value) for value in table.column(cfg.id_column).to_pylist()]
    texts = [value or "" for value in table.column(cfg.text_column).to_pylist()]
    # Stable base order (id-sorted), then a seeded permutation: identical
    # shards regardless of part read order.
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    random.Random(f"{cfg.seed}/shards").shuffle(order)

    tokenizer = ByteTokenizer()
    stream: list[int] = []
    for index in order:
        stream.extend(tokenizer.encode(texts[index]))
        stream.append(ByteTokenizer.EOS)

    step = cfg.seq_len + 1
    n_sequences = len(stream) // step
    sequences = [stream[i * step : (i + 1) * step] for i in range(n_sequences)]
    if not sequences:
        raise ValueError(
            f"silver/{dataset} has too few tokens ({len(stream)}) for seq_len {cfg.seq_len}"
        )

    out_table = pa.table(
        {
            "seq_id": [f"{shards_dataset}-{i:08d}" for i in range(n_sequences)],
            "tokens": pa.array(sequences, type=pa.list_(pa.uint16())),
        }
    )
    shard_parts = catalog.replace_dataset(
        out_table, Layer.GOLD, shards_dataset, cfg.sequences_per_shard
    )

    gold_manifest = build_manifest(catalog, Layer.GOLD, shards_dataset)
    emit_event(
        catalog.root,
        job=f"shard:{dataset}",
        inputs=[
            dataset_ref(f"silver/{dataset}", silver_manifest.content_hash, silver_manifest.n_rows)
        ],
        outputs=[
            dataset_ref(f"gold/{shards_dataset}", gold_manifest.content_hash, gold_manifest.n_rows)
        ],
        facets={"n_sequences": n_sequences, "seq_len": cfg.seq_len},
    )
    snapshot_stage(catalog, "shard", cfg, [silver_manifest], (Layer.GOLD, shards_dataset))

    return ShardResult(
        dataset=dataset,
        shards_dataset=shards_dataset,
        n_docs=len(ids),
        n_sequences=n_sequences,
        n_tokens=n_sequences * step,
        dropped_tail_tokens=len(stream) - n_sequences * step,
        shard_parts=shard_parts,
        vocab_size=ByteTokenizer.vocab_size,
        elapsed_s=round(time.perf_counter() - started, 3),
    )


class ShardIterator:
    """Deterministic shuffled iteration over shard sequences, with exact resume.

    The shuffle is a streaming reservoir: a seeded buffer of
    ``shuffle_buffer`` sequences; each step yields a random buffer slot and
    refills it from the stream. State is ``{"epoch", "consumed"}``;
    ``ShardReader.iterate(resume_state=...)`` rebuilds the iterator and
    replays (discards) the first ``consumed`` yields, which is exact because
    every draw comes from the same seeded RNG stream.
    """

    def __init__(
        self, parts_tables: list[pa.Table], epoch: int, seed: int, shuffle_buffer: int
    ) -> None:
        self.epoch = epoch
        self.consumed = 0
        self._rng = random.Random(f"{seed}/epoch-{epoch}")
        self._incoming = self._sequence_stream(parts_tables)
        self._buffer: list[list[int]] = []
        self._buffer_size = max(1, shuffle_buffer)

    @staticmethod
    def _sequence_stream(parts_tables: list[pa.Table]) -> Iterator[list[int]]:
        for table in parts_tables:
            yield from table.column("tokens").to_pylist()

    def state(self) -> dict[str, int]:
        return {"epoch": self.epoch, "consumed": self.consumed}

    def __iter__(self) -> ShardIterator:
        return self

    def __next__(self) -> list[int]:
        while len(self._buffer) < self._buffer_size:
            try:
                self._buffer.append(next(self._incoming))
            except StopIteration:
                break
        if not self._buffer:
            raise StopIteration
        index = self._rng.randrange(len(self._buffer))
        self._buffer[index], self._buffer[-1] = self._buffer[-1], self._buffer[index]
        sequence = self._buffer.pop()
        self.consumed += 1
        return sequence


class ShardReader:
    """Streaming access to gold shard parts (sorted part order, stable)."""

    def __init__(
        self,
        catalog: Catalog,
        shards_dataset: str,
        seed: int = 0,
        shuffle_buffer: int = 256,
    ) -> None:
        self.catalog = catalog
        self.shards_dataset = shards_dataset
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def _parts_tables(self) -> list[pa.Table]:
        import pyarrow.parquet as pq

        return [pq.read_table(path) for path in self.catalog.parts(Layer.GOLD, self.shards_dataset)]

    def n_sequences(self) -> int:
        return self.catalog.row_count(Layer.GOLD, self.shards_dataset)

    def iterate(self, epoch: int = 0, resume_state: dict[str, int] | None = None) -> ShardIterator:
        if resume_state is not None and resume_state["epoch"] != epoch:
            raise ValueError(
                f"resume state is for epoch {resume_state['epoch']}, requested {epoch}"
            )
        iterator = ShardIterator(self._parts_tables(), epoch, self.seed, self.shuffle_buffer)
        if resume_state is not None:
            for _ in range(resume_state["consumed"]):
                next(iterator)
        return iterator
