"""Each rule must catch the synthetic generator's planted defect strings —
imported from crucible.synth rather than copied, so generator and rules
cannot silently drift apart."""

import pytest

from crucible.quality import RULES, evaluate_text
from crucible.quality.gate import QualityConfig
from crucible.synth import _BOILERPLATE, _MOJIBAKE

CFG = QualityConfig()

CLEAN_SAMPLES = [
    "The regional council in Harborview was approved after weeks of debate. "
    "The quarterly index in Port Anselm was delayed after weeks of debate.",
    "Q: How do I fix a noisy router? A: Before anything else, check the manual first.",
    "Simmer the shallots with miso paste until golden, about 12 minutes.",
]


@pytest.mark.parametrize("text", CLEAN_SAMPLES)
def test_clean_text_passes_all_rules(text: str) -> None:
    all_rules = QualityConfig(rules=sorted(RULES))
    assert evaluate_text(text, all_rules) == []


def test_non_empty_catches_blank() -> None:
    assert "non_empty" in evaluate_text("", CFG)
    assert "non_empty" in evaluate_text("   \n ", CFG)


@pytest.mark.parametrize("text", ["ok", "n/a", "see above", "???", ".", "todo"])
def test_min_words_catches_planted_short_junk(text: str) -> None:
    assert "min_words" in evaluate_text(text, CFG)


@pytest.mark.parametrize("text", _MOJIBAKE)
def test_mojibake_detector_catches_planted_strings(text: str) -> None:
    assert "no_mojibake" in evaluate_text(text, CFG)


@pytest.mark.parametrize("text", _BOILERPLATE)
def test_marker_rule_catches_all_planted_boilerplate(text: str) -> None:
    assert "no_boilerplate_markers" in evaluate_text(text, CFG)


@pytest.mark.parametrize(
    "text",
    [
        "Reach me at jane.doe@example.com for details.",
        "Call (555) 0142 tomorrow morning.",
        "SSN on file: 123-45-6789 per the form.",
    ],
)
def test_pii_rule_catches_planted_patterns(text: str) -> None:
    assert "no_pii" in evaluate_text(text, CFG)


def test_repeated_sentences_is_ratio_based_and_opt_in() -> None:
    assert "no_repeated_sentences" not in CFG.rules  # opt-in by measurement
    cfg = QualityConfig(rules=["no_repeated_sentences"])
    spam = "Buy now and save big. Buy now and save big. Something else entirely."
    assert evaluate_text(spam, cfg) == ["no_repeated_sentences"]
    # One duplicate among many sentences stays under the default ratio.
    mild = ". ".join(f"Sentence number {i} here" for i in range(9)) + ". Sentence number 3 here."
    assert evaluate_text(mild, cfg) == []


def test_unknown_rule_rejected_by_config() -> None:
    with pytest.raises(ValueError, match="unknown rules"):
        QualityConfig(rules=["no_vibes"])
