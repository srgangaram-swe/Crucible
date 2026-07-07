import itertools
import threading
from pathlib import Path

import pytest

from crucible.ingest import (
    BrokerBackpressure,
    InMemoryBroker,
    JsonlSource,
    StreamSource,
    land,
    replay_jsonl,
)
from crucible.storage import Catalog, Layer
from crucible.synth import SynthConfig, generate_corpus, write_jsonl


def test_publish_and_poll_preserves_order() -> None:
    broker = InMemoryBroker()
    for i in range(5):
        broker.publish("t", f"m{i}".encode())
    consumer = broker.consumer("t", "g")
    assert consumer.poll(10, timeout=0.05) == [b"m0", b"m1", b"m2", b"m3", b"m4"]
    assert consumer.poll(10, timeout=0.05) == []


def test_independent_consumer_groups() -> None:
    broker = InMemoryBroker()
    broker.publish("t", b"x")
    assert broker.consumer("t", "g1").poll(10, timeout=0.05) == [b"x"]
    assert broker.consumer("t", "g2").poll(10, timeout=0.05) == [b"x"]


def test_commit_and_resume() -> None:
    broker = InMemoryBroker()
    for i in range(5):
        broker.publish("t", f"m{i}".encode())
    consumer = broker.consumer("t", "g")
    assert len(consumer.poll(3, timeout=0.05)) == 3
    consumer.commit()
    consumer.close()
    # New consumer in the same group resumes from the committed offset.
    resumed = broker.consumer("t", "g")
    assert resumed.poll(10, timeout=0.05) == [b"m3", b"m4"]


def test_uncommitted_messages_are_redelivered() -> None:
    broker = InMemoryBroker()
    for i in range(3):
        broker.publish("t", f"m{i}".encode())
    consumer = broker.consumer("t", "g")
    assert len(consumer.poll(10, timeout=0.05)) == 3
    consumer.close()  # no commit
    assert len(broker.consumer("t", "g").poll(10, timeout=0.05)) == 3


def test_backpressure_raises_on_timeout() -> None:
    broker = InMemoryBroker(capacity=3)
    for _ in range(3):
        broker.publish("t", b"x", timeout=0.05)
    with pytest.raises(BrokerBackpressure, match="backlog at capacity"):
        broker.publish("t", b"overflow", timeout=0.05)


def test_backpressure_releases_when_consumer_commits() -> None:
    broker = InMemoryBroker(capacity=3)
    consumer = broker.consumer("t", "g")
    received: list[bytes] = []

    def drain() -> None:
        while len(received) < 10:
            messages = consumer.poll(2, timeout=0.5)
            received.extend(messages)
            consumer.commit()

    thread = threading.Thread(target=drain)
    thread.start()
    for i in range(10):  # would deadlock without a committing consumer
        broker.publish("t", f"m{i}".encode(), timeout=2.0)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert received == [f"m{i}".encode() for i in range(10)]


def test_stream_source_batches_and_drains() -> None:
    broker = InMemoryBroker()
    for i in range(7):
        broker.publish("t", b'{"id": "r%d"}' % i)
    source = StreamSource(broker, "t", batch_size=3, poll_timeout=0.05)
    tables = list(source.batches())
    assert [t.num_rows for t in tables] == [3, 3, 1]


def test_stream_landing_matches_batch_landing(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(generate_corpus(SynthConfig(seed=6, n_docs=120)), corpus)

    catalog = Catalog(tmp_path / "catalog")
    land(JsonlSource(corpus, batch_size=50), catalog, "via_batch", "jsonl")

    broker = InMemoryBroker()
    assert replay_jsonl(corpus, broker, "corpus") == 120
    stream = StreamSource(broker, "corpus", batch_size=50, poll_timeout=0.05)
    result = land(stream, catalog, "via_stream", "stream")
    assert result.rows_written == 120

    # Same content through either path.
    rows = catalog.query(
        "SELECT (SELECT count(*) FROM bronze_via_batch) AS b, "
        "(SELECT count(*) FROM bronze_via_stream) AS s"
    )
    assert rows == [{"b": 120, "s": 120}]


class CrashAfter:
    """Wraps a source; simulates the landing process dying mid-run.

    The crash check happens BEFORE advancing the inner generator: advancing
    it would let StreamSource commit the previous batch first, which would
    simulate a crash *after* commit instead of before it.
    """

    def __init__(self, inner: StreamSource, crash_at_batch: int) -> None:
        self.inner = inner
        self.crash_at_batch = crash_at_batch

    def batches(self):  # type: ignore[no-untyped-def]
        iterator = self.inner.batches()
        for i in itertools.count():
            if i >= self.crash_at_batch:
                raise RuntimeError("simulated crash before commit")
            try:
                yield next(iterator)
            except StopIteration:
                return


def test_consumer_crash_before_commit_lands_exactly_once(tmp_path: Path) -> None:
    """The exactly-once-ish contract, end to end.

    The crash happens after part 2 is published but before its offsets are
    committed (StreamSource commits a batch when the *next* one is
    requested). Recovery re-polls from the committed offset, producing an
    identically-framed batch whose content hash already landed -> skipped.
    """
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(generate_corpus(SynthConfig(seed=6, n_docs=120)), corpus)
    broker = InMemoryBroker()
    replay_jsonl(corpus, broker, "corpus")
    catalog = Catalog(tmp_path / "catalog")

    crashing = CrashAfter(
        StreamSource(broker, "corpus", group="g", batch_size=50, poll_timeout=0.05),
        crash_at_batch=2,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        land(crashing, catalog, "via_stream", "stream")
    assert catalog.row_count(Layer.BRONZE, "via_stream") == 100  # 2 parts landed

    resumed = StreamSource(broker, "corpus", group="g", batch_size=50, poll_timeout=0.05)
    result = land(resumed, catalog, "via_stream", "stream")
    assert result.parts_skipped == 1  # redelivered batch 2, identical framing
    assert result.parts_written == 1  # the final 20 records
    assert catalog.row_count(Layer.BRONZE, "via_stream") == 120  # no dupes, no loss


def test_producer_side_duplication_is_dedups_job_not_landings(tmp_path: Path) -> None:
    """Documented boundary: re-publishing content with *different* batch
    framing legitimately lands duplicate rows — that is data-level
    duplication, removed by the dedup stage (Phase 3), not by landing."""
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(generate_corpus(SynthConfig(seed=6, n_docs=120)), corpus)
    broker = InMemoryBroker()
    replay_jsonl(corpus, broker, "corpus")
    replay_jsonl(corpus, broker, "corpus")  # publish everything twice
    catalog = Catalog(tmp_path / "catalog")

    stream = StreamSource(broker, "corpus", batch_size=50, poll_timeout=0.05)
    land(stream, catalog, "via_stream", "stream")
    assert catalog.row_count(Layer.BRONZE, "via_stream") == 240
    rows = catalog.query("SELECT count(DISTINCT id) AS ids FROM bronze_via_stream")
    assert rows == [{"ids": 120}]  # the duplication is visible to dedup downstream
