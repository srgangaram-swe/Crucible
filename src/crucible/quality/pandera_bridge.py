"""Optional pandera integration (``.[quality]`` extra).

The native Arrow gate in :mod:`crucible.quality.gate` is the promotion
mechanism (it must run with zero extras). This bridge expresses the same
structural contract as a declarative pandera schema for teams that
standardize on pandera, and doubles as an independent second opinion in
tests: silver output must satisfy it, quarantine must not be required to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    import pandera.pandas as pandera_pandas


def _import_pandera() -> Any:
    try:
        import pandera.pandas as pandera_pandas
    except ImportError as exc:
        raise ImportError(
            "the pandera bridge requires the quality extra; "
            "install with: pip install 'crucible-data[quality]'"
        ) from exc
    return pandera_pandas


def corpus_schema(min_words: int = 1) -> pandera_pandas.DataFrameSchema:
    """Structural contract for a silver text corpus."""
    pandera = _import_pandera()
    return pandera.DataFrameSchema(
        {
            "id": pandera.Column(str, unique=True, nullable=False),
            "text": pandera.Column(
                str,
                nullable=False,
                checks=pandera.Check(
                    lambda s: s.str.split().str.len() >= min_words,
                    name=f"min_words_{min_words}",
                ),
            ),
            "source": pandera.Column(str, nullable=False),
            "timestamp": pandera.Column(str, nullable=False),
        },
        strict=False,  # gt_* evaluation columns and extras may ride along
        coerce=False,
    )


def validate_table(table: pa.Table, min_words: int = 1) -> None:
    """Raise pandera.errors.SchemaError if the table violates the contract."""
    schema = corpus_schema(min_words=min_words)
    schema.validate(table.to_pandas(), lazy=False)
