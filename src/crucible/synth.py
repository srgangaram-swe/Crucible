"""Deterministic synthetic corpus generator.

Crucible's tests and research experiments need data whose defects are
*known*: every duplicate, near-duplicate, junk, and PII record carries a
ground-truth label (``gt_kind``, ``gt_dup_of``) so that dedup
precision/recall, quality-gate hit rates, and mixing ratios can be measured
exactly rather than eyeballed.

Ground-truth fields exist for evaluation only. No pipeline stage may read
them — the pipeline must rediscover defects on its own, and the experiment
harness scores it against the labels.

Generation is fully deterministic given a :class:`SynthConfig` (fixed epoch,
seeded RNG, no wall-clock reads), which is verified by the smoke test.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, model_validator

from crucible.utils.hashing import canonical_json, sha256_texts

# Corpus time range starts at a fixed epoch so runs are reproducible.
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

# Dedicated RNG stream per concern so changing one rate does not perturb the
# text of unrelated records. String seeds hash deterministically across runs
# and platforms (random.seed version 2).
_STREAM_TEXT, _STREAM_TIME, _STREAM_DEFECT = "text", "time", "defect"


class RecordKind(StrEnum):
    """Ground-truth classification of a synthetic record."""

    CLEAN = "clean"
    EXACT_DUP = "exact_dup"
    NEAR_DUP = "near_dup"
    JUNK_EMPTY = "junk_empty"
    JUNK_SHORT = "junk_short"
    JUNK_BOILERPLATE = "junk_boilerplate"
    JUNK_MOJIBAKE = "junk_mojibake"
    PII = "pii"

    @property
    def is_junk(self) -> bool:
        return self.value.startswith("junk_")


@dataclass(frozen=True, slots=True)
class SynthRecord:
    """One corpus record. ``gt_*`` fields are evaluation-only ground truth."""

    id: str
    text: str
    source: str
    timestamp: str  # ISO 8601, UTC
    gt_kind: str
    gt_dup_of: str | None


class SynthConfig(BaseModel):
    """Knobs for corpus generation. Rates are fractions of ``n_docs``."""

    seed: int = 0
    n_docs: int = Field(default=500, ge=10)
    domain_weights: dict[str, float] = Field(
        default={"news": 0.4, "forum_qa": 0.3, "code": 0.2, "recipes": 0.1}
    )
    exact_dup_rate: float = Field(default=0.06, ge=0.0, le=0.3)
    near_dup_rate: float = Field(default=0.10, ge=0.0, le=0.3)
    junk_rate: float = Field(default=0.06, ge=0.0, le=0.3)
    pii_rate: float = Field(default=0.04, ge=0.0, le=0.3)
    time_span_days: int = Field(default=30, ge=1)

    @model_validator(mode="after")
    def _check(self) -> SynthConfig:
        unknown = set(self.domain_weights) - set(_VOCAB)
        if unknown:
            raise ValueError(f"unknown domains {sorted(unknown)}; known: {sorted(_VOCAB)}")
        if not self.domain_weights:
            raise ValueError("domain_weights must not be empty")
        if any(weight <= 0 for weight in self.domain_weights.values()):
            raise ValueError("domain weights must be positive")
        defect_total = self.exact_dup_rate + self.near_dup_rate + self.junk_rate + self.pii_rate
        if defect_total > 0.6:
            raise ValueError(f"defect rates sum to {defect_total:.2f}; must be <= 0.6")
        return self


# ---------------------------------------------------------------------------
# Domain text models: small template grammars with disjoint-enough vocabulary
# that domains are statistically distinguishable (needed for the data-mixing
# ablation and drift detection later).
# ---------------------------------------------------------------------------

_VOCAB: dict[str, dict[str, list[str]]] = {
    "news": {
        "nouns": [
            "council",
            "budget",
            "election",
            "storm",
            "market",
            "policy",
            "committee",
            "economy",
            "transit line",
            "reservoir",
            "port",
            "refinery",
            "ministry",
            "index",
        ],
        "verbs": [
            "approved",
            "delayed",
            "announced",
            "rejected",
            "expanded",
            "suspended",
            "reviewed",
            "criticized",
            "unveiled",
            "postponed",
        ],
        "adjs": [
            "regional",
            "controversial",
            "long-awaited",
            "emergency",
            "quarterly",
            "bipartisan",
            "unexpected",
            "provisional",
            "record-breaking",
        ],
        "places": [
            "Harborview",
            "East Malden",
            "the capital district",
            "Port Anselm",
            "the northern corridor",
            "Greenfield County",
        ],
    },
    "forum_qa": {
        "nouns": [
            "router",
            "spreadsheet",
            "compost bin",
            "guitar amp",
            "sourdough starter",
            "bike chain",
            "thermostat",
            "houseplant",
            "camera lens",
            "backup drive",
        ],
        "verbs": [
            "fix",
            "calibrate",
            "clean",
            "replace",
            "troubleshoot",
            "upgrade",
            "reset",
            "maintain",
            "install",
        ],
        "adjs": [
            "noisy",
            "flaky",
            "secondhand",
            "vintage",
            "stubborn",
            "brand-new",
            "intermittent",
            "leaky",
        ],
        "tips": [
            "check the manual first",
            "unplug it for ten minutes",
            "use a smaller torque",
            "keep a spare on hand",
            "log what you changed",
            "test one variable at a time",
        ],
    },
    "code": {
        "fns": [
            "parse_header",
            "merge_intervals",
            "retry_request",
            "flush_cache",
            "walk_tree",
            "encode_frame",
            "score_batch",
            "split_chunks",
        ],
        "args": ["payload", "items", "cursor", "frame", "batch", "node", "buf"],
        "comments": [
            "handle the empty case before iterating",
            "callers rely on stable ordering here",
            "timeout is in milliseconds, not seconds",
            "this path is hot; avoid extra allocations",
            "upstream may send duplicate keys",
        ],
    },
    "recipes": {
        "ingredients": [
            "shallots",
            "smoked paprika",
            "arborio rice",
            "tahini",
            "leeks",
            "cardamom",
            "miso paste",
            "chickpeas",
            "brown butter",
            "preserved lemon",
        ],
        "verbs": [
            "fold",
            "simmer",
            "whisk",
            "toast",
            "deglaze",
            "caramelize",
            "season",
            "rest",
        ],
        "adjs": [
            "fragrant",
            "golden",
            "silky",
            "tender",
            "glossy",
            "crisp",
            "velvety",
        ],
    },
}

_BOILERPLATE = [
    "Home | About | Contact | Subscribe | Privacy Policy | Terms of Service",
    "Click here to accept cookies and continue to the site. Click here to accept cookies.",
    "Sign up for our newsletter! Sign up for our newsletter! Unsubscribe at any time.",
    "Loading... Loading... Please enable JavaScript to view this page.",
]

# Intentionally garbled text (double-encoded UTF-8 artifacts) for the quality
# gates to catch; the ambiguous-unicode lint is suppressed on purpose.
_MOJIBAKE = [
    "Ã¢â‚¬Å“quarterly reportÃ¢â‚¬Â\x9d shows Ã¢â‚¬â„¢ unexpected Ã‚Â growth Ã¢â‚¬â€œ analysts",  # noqa: RUF001
    "caf�� men� item list ��� encoding test failed",
    "Ã©Ã©Ã© rÃ©sumÃ©Ã‚Â Ã‚Â submitted Ã¢â‚¬â€ pending Ã¢â‚¬Â¦",  # noqa: RUF001
]


def _pick(rng: random.Random, options: list[str]) -> str:
    return options[rng.randrange(len(options))]


def _make_doc(domain: str, rng: random.Random) -> str:
    vocab = _VOCAB[domain]
    n_sentences = rng.randint(3, 9)
    sentences: list[str] = []
    for _ in range(n_sentences):
        if domain == "news":
            sentences.append(
                f"The {_pick(rng, vocab['adjs'])} {_pick(rng, vocab['nouns'])} in "
                f"{_pick(rng, vocab['places'])} was {_pick(rng, vocab['verbs'])} "
                f"after weeks of debate."
            )
        elif domain == "forum_qa":
            sentences.append(
                f"Q: How do I {_pick(rng, vocab['verbs'])} a {_pick(rng, vocab['adjs'])} "
                f"{_pick(rng, vocab['nouns'])}? "
                f"A: Before anything else, {_pick(rng, vocab['tips'])}."
            )
        elif domain == "code":
            fn = _pick(rng, vocab["fns"])
            arg = _pick(rng, vocab["args"])
            sentences.append(
                f"# {_pick(rng, vocab['comments'])}\n"
                f"def {fn}({arg}, limit={rng.randint(2, 64)}):\n"
                f"    return [x for x in {arg} if x is not None][:limit]"
            )
        else:  # recipes
            sentences.append(
                f"{_pick(rng, vocab['verbs']).capitalize()} the "
                f"{_pick(rng, vocab['ingredients'])} with "
                f"{_pick(rng, vocab['ingredients'])} until {_pick(rng, vocab['adjs'])}, "
                f"about {rng.randint(2, 40)} minutes."
            )
    return "\n".join(sentences) if domain == "code" else " ".join(sentences)


def _mutate_near_dup(text: str, rng: random.Random, all_words: list[str]) -> str:
    """Apply small token-level edits (~8-15%) so Jaccard similarity stays high."""
    tokens = text.split()
    if len(tokens) < 4:
        return text + " (edited)"
    n_edits = max(1, round(len(tokens) * rng.uniform(0.08, 0.15)))
    for _ in range(n_edits):
        op = rng.randrange(3)
        pos = rng.randrange(len(tokens))
        if op == 0 and len(tokens) > 4:
            del tokens[pos]
        elif op == 1:
            tokens.insert(pos, _pick(rng, all_words))
        else:
            tokens[pos] = _pick(rng, all_words)
    mutated = " ".join(tokens)
    return mutated if mutated != text else mutated + " (edited)"


def _make_junk(kind: RecordKind, rng: random.Random) -> str:
    if kind is RecordKind.JUNK_EMPTY:
        return ""
    if kind is RecordKind.JUNK_SHORT:
        return _pick(rng, ["ok", "n/a", "see above", "???", ".", "todo"])
    if kind is RecordKind.JUNK_BOILERPLATE:
        return _pick(rng, _BOILERPLATE)
    return _pick(rng, _MOJIBAKE)


def _make_pii_doc(domain: str, rng: random.Random) -> str:
    """A normal document with obviously-synthetic PII spans injected."""
    base = _make_doc(domain, rng)
    email = f"user{rng.randint(100, 999)}@example.com"
    phone = f"(555) 01{rng.randint(10, 99)}"
    return f"{base} Contact me at {email} or call {phone}."


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Draft:
    """A record before time-sorting and id assignment."""

    text: str
    source: str
    ts: datetime
    kind: RecordKind
    dup_of_idx: int | None  # index into the draft list
    order: int  # creation order, tie-breaker for stable sort


def generate_corpus(cfg: SynthConfig) -> list[SynthRecord]:
    """Generate ``cfg.n_docs`` records, time-sorted, ids monotone in time."""
    text_rng = random.Random(f"{cfg.seed}/{_STREAM_TEXT}")
    time_rng = random.Random(f"{cfg.seed}/{_STREAM_TIME}")
    defect_rng = random.Random(f"{cfg.seed}/{_STREAM_DEFECT}")

    n_exact = round(cfg.n_docs * cfg.exact_dup_rate)
    n_near = round(cfg.n_docs * cfg.near_dup_rate)
    n_junk = round(cfg.n_docs * cfg.junk_rate)
    n_pii = round(cfg.n_docs * cfg.pii_rate)
    n_clean = cfg.n_docs - n_exact - n_near - n_junk - n_pii

    domains = sorted(cfg.domain_weights)
    weights = [cfg.domain_weights[d] for d in domains]
    span = timedelta(days=cfg.time_span_days)
    all_words = sorted({w for v in _VOCAB.values() for words in v.values() for w in words})

    def random_ts() -> datetime:
        return _EPOCH + span * time_rng.random()

    drafts: list[_Draft] = []
    for i in range(n_clean):
        domain = text_rng.choices(domains, weights=weights, k=1)[0]
        drafts.append(
            _Draft(_make_doc(domain, text_rng), domain, random_ts(), RecordKind.CLEAN, None, i)
        )

    clean_indices = list(range(n_clean))

    def dup_ts(orig: datetime) -> datetime:
        # Duplicates arrive after their original, still within the span.
        delta = timedelta(seconds=defect_rng.uniform(60, 3 * 86400))
        return min(orig + delta, _EPOCH + span)

    for i in range(n_exact):
        src = drafts[defect_rng.choice(clean_indices)]
        drafts.append(
            _Draft(
                src.text, src.source, dup_ts(src.ts), RecordKind.EXACT_DUP, src.order, n_clean + i
            )
        )
    for i in range(n_near):
        src = drafts[defect_rng.choice(clean_indices)]
        mutated = _mutate_near_dup(src.text, defect_rng, all_words)
        drafts.append(
            _Draft(
                mutated,
                src.source,
                dup_ts(src.ts),
                RecordKind.NEAR_DUP,
                src.order,
                n_clean + n_exact + i,
            )
        )

    junk_kinds = [
        RecordKind.JUNK_EMPTY,
        RecordKind.JUNK_SHORT,
        RecordKind.JUNK_BOILERPLATE,
        RecordKind.JUNK_MOJIBAKE,
    ]
    base_order = n_clean + n_exact + n_near
    for i in range(n_junk):
        kind = junk_kinds[i % len(junk_kinds)]
        domain = defect_rng.choices(domains, weights=weights, k=1)[0]
        drafts.append(
            _Draft(_make_junk(kind, defect_rng), domain, random_ts(), kind, None, base_order + i)
        )

    base_order += n_junk
    for i in range(n_pii):
        domain = defect_rng.choices(domains, weights=weights, k=1)[0]
        drafts.append(
            _Draft(
                _make_pii_doc(domain, defect_rng),
                domain,
                random_ts(),
                RecordKind.PII,
                None,
                base_order + i,
            )
        )

    drafts.sort(key=lambda d: (d.ts, d.order))
    order_to_id = {draft.order: f"rec-{pos:06d}" for pos, draft in enumerate(drafts)}
    return [
        SynthRecord(
            id=f"rec-{pos:06d}",
            text=draft.text,
            source=draft.source,
            timestamp=draft.ts.isoformat(),
            gt_kind=draft.kind.value,
            gt_dup_of=None if draft.dup_of_idx is None else order_to_id[draft.dup_of_idx],
        )
        for pos, draft in enumerate(drafts)
    ]


# ---------------------------------------------------------------------------
# Serialization and reporting
# ---------------------------------------------------------------------------


def corpus_sha256(records: list[SynthRecord]) -> str:
    """Order-sensitive content hash of the corpus (the dataset's identity)."""
    return sha256_texts(canonical_json(asdict(record)) for record in records)


def write_jsonl(records: list[SynthRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_parquet(records: list[SynthRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list[str | None]] = {
        "id": [r.id for r in records],
        "text": [r.text for r in records],
        "source": [r.source for r in records],
        "timestamp": [r.timestamp for r in records],
        "gt_kind": [r.gt_kind for r in records],
        "gt_dup_of": [r.gt_dup_of for r in records],
    }
    pq.write_table(pa.table(columns), path)  # type: ignore[no-untyped-call]


def generation_report(cfg: SynthConfig, records: list[SynthRecord]) -> dict[str, object]:
    by_kind: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for record in records:
        by_kind[record.gt_kind] = by_kind.get(record.gt_kind, 0) + 1
        by_source[record.source] = by_source.get(record.source, 0) + 1
    return {
        "n_records": len(records),
        "seed": cfg.seed,
        "corpus_sha256": corpus_sha256(records),
        "by_kind": dict(sorted(by_kind.items())),
        "by_source": dict(sorted(by_source.items())),
    }
