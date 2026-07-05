from pathlib import Path

from crucible.utils.hashing import canonical_json, sha256_file, sha256_text, sha256_texts


def test_canonical_json_is_key_order_invariant() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert canonical_json({"a": 2, "b": 1}) == '{"a":2,"b":1}'


def test_sha256_texts_is_order_sensitive() -> None:
    assert sha256_texts(["a", "b"]) != sha256_texts(["b", "a"])
    assert sha256_texts(["a", "b"]) == sha256_texts(["a", "b"])


def test_sha256_texts_line_framing() -> None:
    # ["ab"] and ["a", "b"] must not collide.
    assert sha256_texts(["ab"]) != sha256_texts(["a", "b"])


def test_sha256_file_matches_content(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"crucible")
    import hashlib

    assert sha256_file(path) == hashlib.sha256(b"crucible").hexdigest()
    assert sha256_text("crucible") == hashlib.sha256(b"crucible").hexdigest()
