"""Phase 8 data-centric studies using a deterministic byte-bigram proxy model."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from itertools import pairwise
from typing import Any

import numpy as np

from crucible.assay.harness import ExperimentConfig
from crucible.dedup import DedupConfig
from crucible.dedup.pipeline import find_duplicates
from crucible.quality import QualityConfig
from crucible.quality.rules import evaluate_text
from crucible.synth import SynthConfig, generate_corpus

Study = Callable[[ExperimentConfig, int], list[dict[str, Any]]]


def _texts(
    config: ExperimentConfig, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    records = generate_corpus(SynthConfig(seed=seed, n_docs=config.n_docs))
    rows = [
        {"id": record.id, "text": record.text, "source": record.source, "kind": record.gt_kind}
        for record in records
    ]
    clean = [row for row in rows if row["kind"] == "clean"]
    split = max(1, int(len(clean) * 0.75))
    validation = clean[split:]
    validation_ids = {row["id"] for row in validation}
    training = [row for row in rows if row["id"] not in validation_ids]
    return training, validation


def _token_budget(texts: Iterable[str], budget: int) -> bytes:
    documents = [text.encode("utf-8") + b"\xff" for text in texts if text]
    if not documents:
        raise ValueError("study arm contains no trainable text")
    stream = bytearray()
    index = 0
    while len(stream) < budget:
        stream.extend(documents[index % len(documents)])
        index += 1
    return bytes(stream[:budget])


def _bigram_loss(train: bytes, validation: Iterable[str]) -> tuple[float, dict[str, float]]:
    counts = np.ones((256, 256), dtype=np.float64)
    for left, right in pairwise(train):
        counts[left, right] += 1
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    by_domain: dict[str, list[float]] = defaultdict(list)
    all_losses: list[float] = []
    for packed in validation:
        domain, text = packed.split("\0", 1)
        encoded = text.encode("utf-8")
        losses = [-math.log(probabilities[left, right]) for left, right in pairwise(encoded)]
        if losses:
            value = float(np.mean(losses))
            by_domain[domain].append(value)
            all_losses.extend(losses)
    return float(np.mean(all_losses)), {
        domain: float(np.mean(values)) for domain, values in sorted(by_domain.items())
    }


def _evaluate(
    arm: str,
    seed: int,
    train_rows: list[dict[str, str]],
    validation: list[dict[str, str]],
    budget: int,
    **metrics: int | float | str,
) -> dict[str, Any]:
    train = _token_budget((row["text"] for row in train_rows), budget)
    unique_content_tokens = min(
        budget,
        sum(len(text.encode("utf-8")) + 1 for text in {row["text"] for row in train_rows}),
    )
    loss, domains = _bigram_loss(train, (f"{row['source']}\0{row['text']}" for row in validation))
    return {
        "arm": arm,
        "seed": seed,
        "validation_loss": round(loss, 6),
        "train_tokens": len(train),
        "unique_content_tokens": unique_content_tokens,
        "kept_docs": len(train_rows),
        **{f"loss_{domain}": round(value, 6) for domain, value in domains.items()},
        **metrics,
    }


def dedup_ablation(config: ExperimentConfig, seed: int) -> list[dict[str, Any]]:
    rows, validation = _texts(config, seed)
    ids = [row["id"] for row in rows]
    texts = [row["text"] for row in rows]
    thresholds = config.parameters.get("thresholds", [0.4, 0.5, 0.6])
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        duplicates = find_duplicates(
            texts, DedupConfig(threshold=float(threshold), num_perm=32, bands=8), order_keys=ids
        )
        kept = [row for index, row in enumerate(rows) if index not in duplicates.removed]
        results.append(
            _evaluate(
                f"threshold_{float(threshold):.2f}",
                seed,
                kept,
                validation,
                config.train_tokens,
                removed_docs=len(duplicates.removed),
            )
        )
    return results


def quality_ablation(config: ExperimentConfig, seed: int) -> list[dict[str, Any]]:
    rows, validation = _texts(config, seed)
    arms = {
        "ungated": None,
        "default": QualityConfig(),
        "repeated_sentences": QualityConfig(
            rules=[*QualityConfig().rules, "no_repeated_sentences"]
        ),
    }
    results: list[dict[str, Any]] = []
    for arm, quality in arms.items():
        kept = (
            rows
            if quality is None
            else [row for row in rows if not evaluate_text(row["text"], quality)]
        )
        results.append(
            _evaluate(
                arm,
                seed,
                kept,
                validation,
                config.train_tokens,
                keep_rate=round(len(kept) / len(rows), 6),
            )
        )
    return results


def _sample_mixture(
    rows: list[dict[str, str]], weights: dict[str, float], n: int, seed: int
) -> list[dict[str, str]]:
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    rng = random.Random(seed)
    sampled: list[dict[str, str]] = []
    sources = sorted(weights)
    for index in range(n):
        source = rng.choices(sources, weights=[weights[s] for s in sources], k=1)[0]
        sampled.append(by_source[source][index % len(by_source[source])])
    return sampled


def mixture_ablation(config: ExperimentConfig, seed: int) -> list[dict[str, Any]]:
    rows, validation = _texts(config, seed)
    proxy_validation = validation[::2]
    held_out_validation = validation[1::2] or proxy_validation
    sources = sorted({row["source"] for row in rows})
    candidates: dict[str, dict[str, float]] = {
        "uniform": dict.fromkeys(sources, 1.0),
        "observed": {source: sum(row["source"] == source for row in rows) for source in sources},
        "code_heavy": {source: (4.0 if source == "code" else 1.0) for source in sources},
    }
    scored: list[tuple[float, str, list[dict[str, str]]]] = []
    for index, (name, weights) in enumerate(candidates.items()):
        sample = _sample_mixture(rows, weights, len(rows), seed + index)
        proxy = _evaluate(name, seed, sample, proxy_validation, config.train_tokens)
        scored.append((float(proxy["validation_loss"]), name, sample))
    _, best_name, best_sample = min(scored)
    results = [
        _evaluate(name, seed, sample, held_out_validation, config.train_tokens, selection="grid")
        for _, name, sample in scored
    ]
    results.append(
        _evaluate(
            f"proxy_selected_{best_name}",
            seed,
            best_sample,
            held_out_validation,
            config.train_tokens,
            selection="proxy",
        )
    )
    return results


def scaling_law(config: ExperimentConfig, seed: int) -> list[dict[str, Any]]:
    rows, validation = _texts(config, seed)
    ids = [row["id"] for row in rows]
    duplicates = find_duplicates(
        [row["text"] for row in rows],
        DedupConfig(threshold=0.5, num_perm=32, bands=8),
        order_keys=ids,
    )
    deduplicated = [row for index, row in enumerate(rows) if index not in duplicates.removed]
    scales = [
        int(value) for value in config.parameters.get("token_scales", [4000, 8000, 16000, 32000])
    ]
    results = [
        _evaluate(
            f"tokens_{tokens}",
            seed,
            deduplicated,
            validation,
            tokens,
            total_tokens=tokens,
            removed_docs=len(duplicates.removed),
        )
        for tokens in scales
    ]
    slope = float(
        np.polyfit(
            np.log(np.array(scales, dtype=float)),
            np.log(np.array([float(row["validation_loss"]) for row in results])),
            1,
        )[0]
    )
    unique_slope = float(
        np.polyfit(
            np.log(
                np.array(
                    [max(1, int(row["unique_content_tokens"])) for row in results],
                    dtype=float,
                )
            ),
            np.log(np.array([float(row["validation_loss"]) for row in results])),
            1,
        )[0]
    )
    for row in results:
        row["fitted_log_slope"] = round(slope, 6)
        row["fitted_unique_log_slope"] = round(unique_slope, 6)
    return results


STUDIES: dict[str, Study] = {
    "dedup_ablation": dedup_ablation,
    "mixture_ablation": mixture_ablation,
    "quality_ablation": quality_ablation,
    "scaling_law": scaling_law,
}
