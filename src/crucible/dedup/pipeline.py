"""Silver deduplication: exact + near-dup clustering, earliest record kept.

Like the quality gate, dedup is a pure function of (input content, config):
re-running rebuilds identical content-addressed silver parts. The dedup
report (JSON + Markdown, under ``<root>/reports/dedup/``) records cluster
statistics *and the removed record ids*, so evaluation code can score the
decisions afterwards without dedup itself ever reading ground truth.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from crucible.dedup.exact import exact_duplicate_groups, normalize
from crucible.dedup.minhash import MinHasher, jaccard, lsh_candidate_pairs, shingles
from crucible.lineage import dataset_ref, emit_event
from crucible.storage import Catalog, Layer
from crucible.versioning import build_manifest, snapshot_stage


class DedupConfig(BaseModel):
    """Dedup policy; defaults are measured against the synthetic corpus."""

    text_column: str = "text"
    id_column: str = "id"
    exact_normalize: str = "whitespace"  # none | whitespace | aggressive
    shingle_size: int = Field(default=3, ge=1)
    num_perm: int = Field(default=128, ge=2)
    bands: int = Field(default=32, ge=1)
    threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    seed: int = 0
    backend: str = "native"  # native | datasketch
    part_rows: int = Field(default=1000, ge=1)

    @model_validator(mode="after")
    def _check(self) -> DedupConfig:
        if self.num_perm % self.bands != 0:
            raise ValueError(f"bands ({self.bands}) must divide num_perm ({self.num_perm})")
        if self.exact_normalize not in ("none", "whitespace", "aggressive"):
            raise ValueError(f"unknown exact_normalize {self.exact_normalize!r}")
        if self.backend not in ("native", "datasketch"):
            raise ValueError(f"unknown backend {self.backend!r}")
        return self


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            self.parent[rb] = ra


@dataclass(frozen=True, slots=True)
class Duplicates:
    """Clustering outcome over row indices (no storage side effects)."""

    clusters: list[list[int]]  # each ordered by keep-priority; [0] is kept
    removed: list[int]
    removed_exact: int
    removed_near: int
    candidate_pairs: int


def _native_candidates(shingle_sets: list[set[str]], cfg: DedupConfig) -> set[tuple[int, int]]:
    hasher = MinHasher(num_perm=cfg.num_perm, seed=cfg.seed)
    signatures = [hasher.signature(s) for s in shingle_sets]
    return lsh_candidate_pairs(signatures, bands=cfg.bands)


def _datasketch_candidates(shingle_sets: list[set[str]], cfg: DedupConfig) -> set[tuple[int, int]]:
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as exc:
        raise ImportError(
            "backend 'datasketch' requires the dedup extra; "
            "install with: pip install 'crucible-data[dedup]'"
        ) from exc
    # Explicit banding to match the native backend's LSH geometry —
    # datasketch's threshold-based auto-tuning would otherwise pick a more
    # conservative S-curve and silently lose candidate recall.
    lsh = MinHashLSH(num_perm=cfg.num_perm, params=(cfg.bands, cfg.num_perm // cfg.bands))
    sketches = []
    for index, shingle_set in enumerate(shingle_sets):
        sketch = MinHash(num_perm=cfg.num_perm, seed=cfg.seed)
        for shingle in shingle_set:
            sketch.update(shingle.encode("utf-8"))
        sketches.append(sketch)
        lsh.insert(str(index), sketch)
    pairs: set[tuple[int, int]] = set()
    for index, sketch in enumerate(sketches):
        for key in lsh.query(sketch):
            other = int(key)
            if other != index:
                pairs.add((min(index, other), max(index, other)))
    return pairs


def find_duplicates(
    texts: list[str], cfg: DedupConfig, order_keys: list[str] | None = None
) -> Duplicates:
    """Cluster exact and near duplicates over a list of texts.

    ``order_keys`` decides which cluster member survives: the record with
    the smallest key is kept. Callers must pass a key that is stable across
    storage row order — ``run_dedup`` passes the id column, which for the
    synthetic corpus is monotone in event time (earliest record wins).
    Relying on row position instead would be a bug: Parquet parts are
    content-hash-named, so ``Catalog.read`` order is not ingestion order.
    """
    n = len(texts)
    uf = _UnionFind(n)

    for group in exact_duplicate_groups(texts, cfg.exact_normalize):
        for index in group[1:]:
            uf.union(group[0], index)

    shingle_sets = [shingles(text, cfg.shingle_size) for text in texts]
    if cfg.backend == "native":
        candidates = _native_candidates(shingle_sets, cfg)
    else:
        candidates = _datasketch_candidates(shingle_sets, cfg)

    for first, second in sorted(candidates):
        if uf.find(first) != uf.find(second) and (
            jaccard(shingle_sets[first], shingle_sets[second]) >= cfg.threshold
        ):
            uf.union(first, second)

    def keep_priority(index: int) -> tuple[str, int]:
        return (order_keys[index] if order_keys is not None else "", index)

    members: dict[int, list[int]] = {}
    for index in range(n):
        members.setdefault(uf.find(index), []).append(index)
    clusters = sorted(
        (sorted(group, key=keep_priority) for group in members.values() if len(group) > 1),
        key=lambda group: keep_priority(group[0]),
    )

    removed: list[int] = []
    removed_exact = removed_near = 0
    for cluster in clusters:
        kept_norm = normalize(texts[cluster[0]], cfg.exact_normalize)
        for index in cluster[1:]:
            removed.append(index)
            if normalize(texts[index], cfg.exact_normalize) == kept_norm:
                removed_exact += 1
            else:
                removed_near += 1

    return Duplicates(
        clusters=clusters,
        removed=sorted(removed),
        removed_exact=removed_exact,
        removed_near=removed_near,
        candidate_pairs=len(candidates),
    )


@dataclass(frozen=True, slots=True)
class DedupResult:
    dataset: str
    input_rows: int
    kept_rows: int
    removed_exact: int
    removed_near: int
    n_clusters: int
    largest_cluster: int
    candidate_pairs: int
    removed_ids: list[str]
    elapsed_s: float
    silver_parts: int
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_dedup(catalog: Catalog, dataset: str, cfg: DedupConfig) -> DedupResult:
    """Deduplicate silver/<dataset> in place (earliest record per cluster kept)."""
    started = time.perf_counter()
    pre_manifest = build_manifest(catalog, Layer.SILVER, dataset)
    table = catalog.read(Layer.SILVER, dataset)
    texts = [value or "" for value in table.column(cfg.text_column).to_pylist()]
    ids = [str(value) for value in table.column(cfg.id_column).to_pylist()]

    duplicates = find_duplicates(texts, cfg, order_keys=ids)
    removed_set = set(duplicates.removed)
    kept_indices = [index for index in range(table.num_rows) if index not in removed_set]
    silver_parts = catalog.replace_dataset(
        table.take(kept_indices), Layer.SILVER, dataset, cfg.part_rows
    )

    post_manifest = build_manifest(catalog, Layer.SILVER, dataset)
    emit_event(
        catalog.root,
        job=f"dedup:{dataset}",
        inputs=[dataset_ref(f"silver/{dataset}", pre_manifest.content_hash, pre_manifest.n_rows)],
        outputs=[
            dataset_ref(f"silver/{dataset}", post_manifest.content_hash, post_manifest.n_rows)
        ],
        facets={
            "removed_exact": duplicates.removed_exact,
            "removed_near": duplicates.removed_near,
        },
    )
    snapshot_stage(catalog, "dedup", cfg, [pre_manifest], (Layer.SILVER, dataset))

    return DedupResult(
        dataset=dataset,
        input_rows=table.num_rows,
        kept_rows=len(kept_indices),
        removed_exact=duplicates.removed_exact,
        removed_near=duplicates.removed_near,
        n_clusters=len(duplicates.clusters),
        largest_cluster=max((len(c) for c in duplicates.clusters), default=1),
        candidate_pairs=duplicates.candidate_pairs,
        removed_ids=[ids[index] for index in duplicates.removed],
        elapsed_s=round(time.perf_counter() - started, 3),
        silver_parts=silver_parts,
        config=cfg.model_dump(mode="json"),
    )


def write_dedup_report(result: DedupResult, root: Path) -> tuple[Path, Path]:
    report_dir = root / "reports" / "dedup"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{result.dataset}.json"
    md_path = report_dir / f"{result.dataset}.md"
    json_path.write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    removed = result.removed_exact + result.removed_near
    rate = removed / result.input_rows if result.input_rows else 0.0
    md_path.write_text(
        "\n".join(
            [
                f"# Dedup report: `{result.dataset}`",
                "",
                f"{result.input_rows} rows in, **{result.kept_rows} kept**, "
                f"{removed} removed ({rate:.1%}: {result.removed_exact} exact, "
                f"{result.removed_near} near) across {result.n_clusters} clusters "
                f"(largest {result.largest_cluster}), {result.candidate_pairs} LSH "
                f"candidate pairs verified, in {result.elapsed_s}s.",
                "",
                f"Backend `{result.config['backend']}`, threshold "
                f"{result.config['threshold']}, {result.config['num_perm']} permutations x "
                f"{result.config['bands']} bands, shingle size "
                f"{result.config['shingle_size']}.",
                "",
                "_Removed ids are recorded in the JSON report for evaluation-only scoring._",
                "",
            ]
        )
    )
    return json_path, md_path
