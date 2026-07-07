"""Optional integration test for the real Kafka transport.

Runs only when both confluent-kafka is installed (.[stream] extra) and a
broker is reachable at $CRUCIBLE_KAFKA_BOOTSTRAP (e.g. via
docker-compose --profile stream up). The default suite covers identical
semantics through InMemoryBroker.
"""

import json
import os
import uuid
from pathlib import Path

import pytest

pytest.importorskip("confluent_kafka")

BOOTSTRAP = os.environ.get("CRUCIBLE_KAFKA_BOOTSTRAP")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        BOOTSTRAP is None, reason="set CRUCIBLE_KAFKA_BOOTSTRAP to run Kafka integration"
    ),
]


def test_kafka_round_trip_and_landing(tmp_path: Path) -> None:
    from crucible.ingest import KafkaBroker, StreamSource, land
    from crucible.storage import Catalog

    assert BOOTSTRAP is not None
    broker = KafkaBroker(BOOTSTRAP)
    topic = f"crucible-test-{uuid.uuid4().hex[:8]}"
    for i in range(20):
        broker.publish(topic, json.dumps({"id": f"r{i}", "text": f"doc {i}"}).encode())
    broker.flush()

    catalog = Catalog(tmp_path / "catalog")
    source = StreamSource(broker, topic, group=f"g-{uuid.uuid4().hex[:8]}", poll_timeout=5.0)
    result = land(source, catalog, "kafka_demo", "kafka")
    assert result.rows_written == 20
