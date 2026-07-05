import numpy as np
import pytest

from crucible.dedup import MinHasher, jaccard, lsh_candidate_pairs, shingles
from crucible.dedup.exact import exact_duplicate_groups, normalize
from crucible.dedup.minhash import estimate_jaccard

DOC = "the quick brown fox jumps over the lazy dog near the river bank today"


def test_shingles_word_ngrams() -> None:
    assert shingles("a b c d", size=3) == {"a b c", "b c d"}
    assert shingles("a b", size=3) == {"a b"}  # short text collapses
    assert shingles("", size=3) == set()


def test_jaccard_basics() -> None:
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_signatures_are_deterministic_across_instances() -> None:
    doc_shingles = shingles(DOC)
    sig_a = MinHasher(num_perm=64, seed=0).signature(doc_shingles)
    sig_b = MinHasher(num_perm=64, seed=0).signature(doc_shingles)
    assert np.array_equal(sig_a, sig_b)
    sig_c = MinHasher(num_perm=64, seed=1).signature(doc_shingles)
    assert not np.array_equal(sig_a, sig_c)


def test_signature_estimates_jaccard() -> None:
    hasher = MinHasher(num_perm=256, seed=0)
    a = shingles(DOC)
    b = shingles(DOC + " with a small tail appended here")
    true_j = jaccard(a, b)
    est = estimate_jaccard(hasher.signature(a), hasher.signature(b))
    assert est == pytest.approx(true_j, abs=0.12)  # 256 perms -> ~1/16 std error
    assert estimate_jaccard(hasher.signature(a), hasher.signature(a)) == 1.0


def test_lsh_identical_docs_always_collide() -> None:
    hasher = MinHasher(num_perm=128, seed=0)
    sig = hasher.signature(shingles(DOC))
    other = hasher.signature(shingles("completely different words about cooking rice"))
    pairs = lsh_candidate_pairs([sig, sig.copy(), other], bands=32)
    assert (0, 1) in pairs
    assert (0, 2) not in pairs


def test_lsh_bands_must_divide_num_perm() -> None:
    with pytest.raises(ValueError, match="must divide"):
        lsh_candidate_pairs([np.zeros(10, dtype=np.uint64)], bands=3)


def test_exact_groups_and_normalization() -> None:
    texts = ["a  b", "a b", "A B", "c"]
    assert exact_duplicate_groups(texts, "none") == []
    assert exact_duplicate_groups(texts, "whitespace") == [[0, 1]]
    assert exact_duplicate_groups(texts, "aggressive") == [[0, 1, 2]]
    with pytest.raises(ValueError, match="unknown normalize"):
        normalize("x", "vibes")
