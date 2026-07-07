"""Row-level text validation rules.

Heuristics follow the web-corpus filtering literature — boilerplate marker
lists and repeated-span removal as in C4 (Raffel et al., 2020,
arXiv:1910.10683) and RefinedWeb (Penedo et al., 2023, arXiv:2306.01116) —
scaled down to rules whose hit rates we can verify exactly against the
synthetic corpus's planted defects.

Each rule is a pure predicate ``(text, config) -> passed``. The registry
maps rule names (as referenced in QualityConfig.rules) to implementations;
``evaluate_text`` returns the names of every rule a record fails, which
become its quarantine reasons.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crucible.quality.gate import QualityConfig

# Web-chrome phrases that indicate scraped navigation/consent boilerplate
# rather than content (C4-style blocklist, tuned to be domain-neutral).
_BOILERPLATE_MARKERS = (
    "click here",
    "accept cookies",
    "privacy policy",
    "terms of service",
    "sign up for our newsletter",
    "unsubscribe",
    "enable javascript",
    "loading...",
    "all rights reserved",
    "subscribe",
)

# Mojibake: UTF-8 read as latin-1/cp1252 leaves signature digraph sequences
# (the "A-tilde + punctuation" pairs matched below); U+FFFD means a decoder
# already gave up. The pattern intentionally contains those garbled bytes.
_MOJIBAKE_PATTERN = re.compile("�|Ã[¢‚Â©â]|â‚¬|Ã¢â")  # noqa: RUF001

_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_PATTERN = re.compile(r"\(\d{3}\)\s*\d{2,4}|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_SENTENCE_SPLIT = re.compile(r"[.!?]+\s+|\n+")


def non_empty(text: str, cfg: QualityConfig) -> bool:
    return bool(text.strip())


def min_words(text: str, cfg: QualityConfig) -> bool:
    return len(text.split()) >= cfg.min_words


def no_mojibake(text: str, cfg: QualityConfig) -> bool:
    if _MOJIBAKE_PATTERN.search(text):
        return False
    # Unassigned/control-heavy text is another decoder-failure signature.
    controls = sum(1 for ch in text if unicodedata.category(ch) in ("Cc", "Cn")) - text.count("\n")
    return controls == 0


def no_boilerplate_markers(text: str, cfg: QualityConfig) -> bool:
    lowered = text.lower()
    hits = sum(1 for marker in _BOILERPLATE_MARKERS if marker in lowered)
    return hits < cfg.boilerplate_marker_threshold


def no_repeated_sentences(text: str, cfg: QualityConfig) -> bool:
    """Fails when the duplicated-sentence fraction exceeds the configured
    ratio (C4-style repeated-span removal).

    Deliberately NOT in the default rule set: measured on the synthetic
    corpus it adds zero recall (the marker rule already catches every
    planted boilerplate record) while quarantining ~30% of clean
    template-repetitive text. Quantifying exactly that kind of
    aggressiveness/keep-rate tradeoff is the Phase 8 quality ablation.
    """
    sentences = [s.strip().lower() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) < 3:
        return True
    duplicated = len(sentences) - len(set(sentences))
    return duplicated / len(sentences) <= cfg.max_duplicate_sentence_ratio


def no_pii(text: str, cfg: QualityConfig) -> bool:
    return not (
        _EMAIL_PATTERN.search(text) or _PHONE_PATTERN.search(text) or _SSN_PATTERN.search(text)
    )


RULES: dict[str, Callable[[str, QualityConfig], bool]] = {
    "non_empty": non_empty,
    "min_words": min_words,
    "no_mojibake": no_mojibake,
    "no_boilerplate_markers": no_boilerplate_markers,
    "no_repeated_sentences": no_repeated_sentences,
    "no_pii": no_pii,
}


def evaluate_text(text: str, cfg: QualityConfig) -> list[str]:
    """Names of every enabled rule this text fails (empty list = promote)."""
    return [name for name in cfg.rules if not RULES[name](text, cfg)]
