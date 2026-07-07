import pyarrow as pa
import pytest

from crucible.features import (
    FeatureStore,
    LeakageError,
    assert_no_leakage,
    source_rollup_features,
)
from crucible.storage import Catalog

FEATURES = pa.Table.from_pylist(
    [
        {"source": "news", "timestamp": "2026-01-01T00:00:00+00:00", "score": 1},
        {"source": "news", "timestamp": "2026-01-03T00:00:00+00:00", "score": 3},
        {"source": "code", "timestamp": "2026-01-02T00:00:00+00:00", "score": 2},
    ]
)


@pytest.fixture
def store(tmp_path) -> FeatureStore:  # type: ignore[no-untyped-def]
    store = FeatureStore(Catalog(tmp_path / "catalog"))
    store.register("src_stats", FEATURES, "source", "timestamp")
    return store


def test_point_in_time_join_uses_only_the_past(store: FeatureStore) -> None:
    """The core PIT guarantee: a spine row between two feature updates gets
    the earlier one; a naive latest-value join would leak score=3."""
    spine = pa.Table.from_pylist(
        [
            {"id": "a", "source": "news", "timestamp": "2026-01-02T00:00:00+00:00"},
            {"id": "b", "source": "news", "timestamp": "2026-01-04T00:00:00+00:00"},
            {"id": "c", "source": "code", "timestamp": "2026-01-01T12:00:00+00:00"},
        ]
    )
    joined = store.point_in_time_join(spine, "src_stats", "source", "timestamp")
    by_id = {row["id"]: row for row in joined.to_pylist()}
    assert by_id["a"]["src_stats__score"] == 1  # not the future value 3
    assert by_id["b"]["src_stats__score"] == 3  # update now in the past
    assert by_id["c"]["src_stats__score"] is None  # nothing known yet


def test_join_at_exact_timestamp_is_inclusive(store: FeatureStore) -> None:
    spine = pa.Table.from_pylist(
        [{"id": "a", "source": "news", "timestamp": "2026-01-03T00:00:00+00:00"}]
    )
    joined = store.point_in_time_join(spine, "src_stats", "source", "timestamp")
    assert joined.to_pylist()[0]["src_stats__score"] == 3


def test_leakage_guard_catches_corrupted_join() -> None:
    corrupted = pa.Table.from_pylist(
        [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "f__feature_ts": "2026-02-01T00:00:00+00:00",  # from the future
            }
        ]
    )
    with pytest.raises(LeakageError, match="postdates"):
        assert_no_leakage(corrupted, "timestamp", "f__feature_ts")


def test_offline_online_parity(store: FeatureStore) -> None:
    """get_latest must equal the PIT join evaluated arbitrarily far in the
    future — the offline and online stores answer from the same truth."""
    far_future = pa.Table.from_pylist(
        [{"id": "x", "source": "news", "timestamp": "2100-01-01T00:00:00+00:00"}]
    )
    joined = store.point_in_time_join(far_future, "src_stats", "source", "timestamp")
    offline_score = joined.to_pylist()[0]["src_stats__score"]
    online = store.get_latest("src_stats", "news")
    assert online is not None
    assert online["score"] == offline_score == 3
    assert store.get_latest("src_stats", "recipes") is None


def test_register_validates_columns(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = FeatureStore(Catalog(tmp_path))
    with pytest.raises(ValueError, match="missing column"):
        store.register("bad", pa.table({"x": [1]}), "source", "timestamp")
    with pytest.raises(ValueError, match="no feature columns"):
        store.register(
            "bad",
            pa.table({"source": ["a"], "timestamp": ["t"]}),
            "source",
            "timestamp",
        )


def test_view_round_trip_and_listing(store: FeatureStore) -> None:
    view, table = store.load_view("src_stats")
    assert view.feature_columns == ["score"]
    assert table.num_rows == 3
    assert store.views() == ["src_stats"]
    with pytest.raises(FileNotFoundError):
        store.load_view("nope")


def test_source_rollups_are_cumulative_and_pit_safe() -> None:
    corpus = pa.Table.from_pylist(
        [
            {"source": "news", "timestamp": "2026-01-01", "text": "one two three"},
            {"source": "news", "timestamp": "2026-01-02", "text": "one two three four five"},
            {"source": "code", "timestamp": "2026-01-03", "text": "x = 1"},
        ]
    )
    rollups = source_rollup_features(corpus).to_pylist()
    news = [row for row in rollups if row["source"] == "news"]
    assert [row["docs_so_far"] for row in news] == [1, 2]
    assert news[0]["mean_words_so_far"] == 3.0
    assert news[1]["mean_words_so_far"] == 4.0  # (3+5)/2, running mean
