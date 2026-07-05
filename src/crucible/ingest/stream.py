"""Streaming ingestion: broker protocol, in-memory fallback, Kafka adapter.

The fallback ladder in action: :class:`InMemoryBroker` implements the same
``Broker`` protocol as :class:`KafkaBroker` (confluent-kafka, ``.[stream]``
extra), so every streaming code path — including backpressure and
commit/replay semantics — is exercised offline by the default test suite,
and the Kafka adapter only swaps the transport.

Delivery contract: at-least-once. Consumers commit *after* the lander has
published a batch; a crash between publish and commit causes redelivery,
which the content-addressed lander skips (see :mod:`crucible.ingest.land`).
That pairing is what makes the pipeline exactly-once-ish end to end — with
one precise boundary: hash-skipping collapses redeliveries with *identical
framing* (consumer restarts re-poll from the committed offset, so framing
is preserved). Producer-side re-publication frames batches differently and
legitimately lands duplicate rows; that is data-level duplication, removed
by the dedup stage, not a delivery bug.

Scope notes (honest): the in-memory broker models a single-partition topic
with one active consumer per group; there is no rebalancing. That is the
right fidelity for testing landing semantics, not a message bus.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Condition
from typing import TYPE_CHECKING, Protocol

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Iterator


class BrokerBackpressure(Exception):
    """Publish timed out because the slowest consumer group is too far behind."""


class Consumer(Protocol):
    def poll(self, max_messages: int, timeout: float) -> list[bytes]: ...
    def commit(self) -> None: ...
    def close(self) -> None: ...


class Broker(Protocol):
    def publish(self, topic: str, payload: bytes, timeout: float | None = None) -> None: ...
    def consumer(self, topic: str, group: str) -> Consumer: ...
    def flush(self) -> None: ...


# ---------------------------------------------------------------------------
# In-memory fallback
# ---------------------------------------------------------------------------


class InMemoryBroker:
    """Bounded, thread-safe, single-process broker.

    Backpressure semantics: a topic retains at most ``capacity`` messages
    beyond the slowest registered group's committed offset (or beyond
    nothing, if no group is registered). ``publish`` blocks until space
    frees up or ``timeout`` expires, then raises
    :class:`BrokerBackpressure` — producers slow to the slowest consumer,
    they don't silently drop.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self._cond = Condition()
        self._logs: dict[str, list[bytes]] = {}
        self._committed: dict[tuple[str, str], int] = {}
        self._positions: dict[tuple[str, str], int] = {}

    def _backlog(self, topic: str) -> int:
        log_len = len(self._logs.get(topic, []))
        committed = [offset for (t, _g), offset in self._committed.items() if t == topic]
        return log_len - min(committed) if committed else log_len

    def publish(self, topic: str, payload: bytes, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._backlog(topic) >= self.capacity:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise BrokerBackpressure(
                        f"topic {topic!r} backlog at capacity ({self.capacity})"
                    )
                self._cond.wait(remaining)
            self._logs.setdefault(topic, []).append(payload)
            self._cond.notify_all()

    def consumer(self, topic: str, group: str) -> Consumer:
        key = (topic, group)
        with self._cond:
            self._committed.setdefault(key, 0)
            # A (re)created consumer resumes from the committed offset:
            # anything delivered-but-uncommitted is redelivered.
            self._positions[key] = self._committed[key]
        return _InMemoryConsumer(self, key)

    def flush(self) -> None:  # everything is already "delivered"
        return

    # Internal, called by _InMemoryConsumer under self._cond's lock protocol.

    def _poll(self, key: tuple[str, str], max_messages: int, timeout: float) -> list[bytes]:
        topic = key[0]
        deadline = time.monotonic() + timeout
        with self._cond:
            log = self._logs.setdefault(topic, [])
            while self._positions[key] >= len(log):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._cond.wait(remaining)
            position = self._positions[key]
            messages = log[position : position + max_messages]
            self._positions[key] = position + len(messages)
            return messages

    def _commit(self, key: tuple[str, str]) -> None:
        with self._cond:
            self._committed[key] = self._positions[key]
            self._cond.notify_all()


class _InMemoryConsumer:
    def __init__(self, broker: InMemoryBroker, key: tuple[str, str]) -> None:
        self._broker = broker
        self._key = key

    def poll(self, max_messages: int, timeout: float) -> list[bytes]:
        return self._broker._poll(self._key, max_messages, timeout)

    def commit(self) -> None:
        self._broker._commit(self._key)

    def close(self) -> None:
        return


# ---------------------------------------------------------------------------
# Kafka adapter (.[stream] extra; covered by the optional integration test,
# see tests/test_kafka_optional.py)
# ---------------------------------------------------------------------------


class KafkaBroker:  # pragma: no cover - requires a running broker
    """confluent-kafka transport behind the same Broker protocol."""

    def __init__(self, bootstrap_servers: str) -> None:
        try:
            import confluent_kafka
        except ImportError as exc:
            raise ImportError(
                "KafkaBroker requires confluent-kafka; "
                "install with: pip install 'crucible-data[stream]'"
            ) from exc
        self._kafka = confluent_kafka
        self._producer = confluent_kafka.Producer({"bootstrap.servers": bootstrap_servers})
        self._bootstrap = bootstrap_servers

    def publish(self, topic: str, payload: bytes, timeout: float | None = None) -> None:
        # confluent-kafka's own queue provides the backpressure here: produce
        # raises BufferError when the local queue is full.
        try:
            self._producer.produce(topic, payload)
        except BufferError as exc:
            raise BrokerBackpressure(str(exc)) from exc
        self._producer.poll(0)

    def flush(self) -> None:
        self._producer.flush()

    def consumer(self, topic: str, group: str) -> Consumer:
        kafka_consumer = self._kafka.Consumer(
            {
                "bootstrap.servers": self._bootstrap,
                "group.id": group,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        kafka_consumer.subscribe([topic])
        return _KafkaConsumer(kafka_consumer)


class _KafkaConsumer:  # pragma: no cover - requires a running broker
    def __init__(self, consumer: object) -> None:
        self._consumer = consumer

    def poll(self, max_messages: int, timeout: float) -> list[bytes]:
        messages = self._consumer.consume(num_messages=max_messages, timeout=timeout)  # type: ignore[attr-defined]
        return [m.value() for m in messages if m.error() is None]

    def commit(self) -> None:
        self._consumer.commit(asynchronous=False)  # type: ignore[attr-defined]

    def close(self) -> None:
        self._consumer.close()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Broker -> Source adapter, and replay
# ---------------------------------------------------------------------------


class StreamSource:
    """Adapts a broker consumer to the batch ``Source`` protocol.

    Drains the topic until a poll comes back empty (idle for
    ``poll_timeout``), yielding micro-batches of JSON records. The commit
    for a batch happens when the *next* batch is requested — i.e. after the
    caller (the lander) has fully processed the previous one.
    """

    def __init__(
        self,
        broker: Broker,
        topic: str,
        group: str = "crucible",
        batch_size: int = 1000,
        poll_timeout: float = 0.2,
    ) -> None:
        self.broker = broker
        self.topic = topic
        self.group = group
        self.batch_size = batch_size
        self.poll_timeout = poll_timeout

    def batches(self) -> Iterator[pa.Table]:
        consumer = self.broker.consumer(self.topic, self.group)
        try:
            while True:
                payloads = consumer.poll(self.batch_size, self.poll_timeout)
                if not payloads:
                    return
                rows = [json.loads(payload) for payload in payloads]
                yield pa.Table.from_pylist(rows)
                consumer.commit()
        finally:
            consumer.close()


def replay_jsonl(path: Path, broker: Broker, topic: str, timeout: float | None = None) -> int:
    """Publish a JSONL file line-by-line (used to drive the stream path
    offline, e.g. in the smoke test); returns the message count."""
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                broker.publish(topic, line.strip().encode("utf-8"), timeout=timeout)
                count += 1
    broker.flush()
    return count
