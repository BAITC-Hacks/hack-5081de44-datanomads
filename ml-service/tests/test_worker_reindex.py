from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from worker import reindex_qdrant


class FakeResponse:
    def __init__(self, status_code: int, body: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "fake HTTP error",
                request=httpx.Request("GET", "http://fake"),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict[str, Any]:
        return self._body


class FakeQdrantAndMlClient:
    def __init__(
        self,
        events: list[str],
        embedder_version: str,
        fail_on_embedding_call: int | None = None,
        metadata_dimension: int = 8,
    ) -> None:
        self.events = events
        self.embedder_version = embedder_version
        self.fail_on_embedding_call = fail_on_embedding_call
        self.metadata_dimension = metadata_dimension
        self.embedding_calls = 0
        self.collections: dict[str, dict[str, Any]] = {}
        self.points: dict[str, list[dict[str, Any]]] = {}
        self.deleted_collections: list[str] = []

    async def __aenter__(self) -> FakeQdrantAndMlClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def get(self, url: str) -> FakeResponse:
        if url.endswith("/internal/v1/models/embedder"):
            return FakeResponse(
                200,
                {
                    "model_version": self.embedder_version,
                    "dimension": self.metadata_dimension,
                    "distance_metric": "cosine",
                },
            )
        collection = url.rsplit("/", 1)[-1]
        if collection not in self.collections:
            return FakeResponse(404)
        return FakeResponse(
            200,
            {"result": {"config": {"params": {"vectors": self.collections[collection]}}}},
        )

    async def put(self, url: str, json: dict[str, Any]) -> FakeResponse:
        collection = url.split("/collections/", 1)[1].split("/", 1)[0]
        if "/points" in url:
            self.events.append(f"qdrant-upsert:{collection}")
            self.points.setdefault(collection, []).extend(json["points"])
            return FakeResponse(200)
        self.events.append(f"qdrant-create:{collection}")
        vector_config = json["vectors"]
        self.collections[collection] = {
            "size": vector_config["size"],
            "distance": vector_config["distance"],
        }
        return FakeResponse(200)

    async def post(self, url: str, json: dict[str, Any]) -> FakeResponse:
        assert url.endswith("/internal/v1/embed")
        self.embedding_calls += 1
        dimension = json["dimension"]
        model_version = (
            "wrong-embedder"
            if self.embedding_calls == self.fail_on_embedding_call
            else self.embedder_version
        )
        return FakeResponse(
            200,
            {
                "model_version": model_version,
                "dimension": dimension,
                "embeddings": [[0.1] * dimension for _ in json["texts"]],
            },
        )

    async def delete(self, url: str) -> FakeResponse:
        self.deleted_collections.append(url)
        return FakeResponse(200)


class FakeTransaction:
    async def __aenter__(self) -> FakeTransaction:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None


class FakeConnection:
    def __init__(self, pool: FakePool) -> None:
        self.pool = pool

    def transaction(self) -> FakeTransaction:
        return FakeTransaction()

    async def fetchrow(self, query: str) -> dict[str, Any]:
        assert "FOR UPDATE" in query
        return self.pool.active_index

    async def fetch(self, query: str, cursor: int) -> list[dict[str, Any]]:
        assert "ORDER BY id LIMIT 32" in query
        return [row for row in self.pool.rows if int(row["id"]) > cursor]

    async def fetchval(self, query: str) -> int:
        assert "COUNT(*)" in query
        return len(self.pool.rows)

    async def execute(self, query: str, *_: Any) -> None:
        if query.startswith("UPDATE tickets"):
            self.pool.events.append("postgres-ticket-lineage-switch")
        elif query.startswith("UPDATE vector_index_state"):
            self.pool.events.append("postgres-pointer-switch")


class FakePool:
    def __init__(self, events: list[str], add_ticket_during_build: bool = False) -> None:
        self.events = events
        self.add_ticket_during_build = add_ticket_during_build
        self.ticket_added = False
        self.rows = [
            self._ticket_row(ticket_id)
            for ticket_id in (1, 2)
        ]
        self.active_index = {
            "embedder_version": "old-embedder",
            "embedding_dimension": 8,
            "distance_metric": "Cosine",
            "collection_name": "active-old",
            "generation": 3,
        }

    async def fetchval(self, query: str) -> int:
        assert "MAX(id)" in query
        return max((int(row["id"]) for row in self.rows), default=0)

    async def fetch(self, query: str, cursor: int, *upper_bound: int) -> list[dict[str, Any]]:
        if upper_bound:
            batch = [
                row
                for row in self.rows
                if cursor < int(row["id"]) <= upper_bound[0]
            ][:32]
            if self.add_ticket_during_build and not self.ticket_added:
                self.rows.append(self._ticket_row(3))
                self.ticket_added = True
            return batch
        return []

    @staticmethod
    def _ticket_row(ticket_id: int) -> dict[str, Any]:
        return {
            "id": ticket_id,
            "original_text": f"synthetic ticket {ticket_id}",
            "topic_id": "water_supply",
            "region_id": "KZ-ASTANA",
            "created_at": datetime(2026, 9, 27, tzinfo=UTC),
        }

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self)


class FakeAcquire:
    def __init__(self, pool: FakePool) -> None:
        self.connection = FakeConnection(pool)

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(self, *_: Any) -> None:
        return None


def test_reindex_builds_staging_collection_before_switching_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    client = FakeQdrantAndMlClient(events, "new-embedder")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: client)
    pool = FakePool(events, add_ticket_during_build=True)

    result = asyncio.run(
        reindex_qdrant(
            pool,
            {
                "embedder_version": "new-embedder",
                "embedding_dimension": 8,
                "collection_base": "pulse109_new_embedder",
            },
            job_id=42,
        )
    )

    staging_collection = "pulse109_new_embedder_job_42"
    assert result["collection"] == staging_collection
    assert result["previous_collection"] == "active-old"
    assert result["indexed_rows"] == 3
    assert [point["id"] for point in client.points[staging_collection]] == [1, 2, 3]
    assert not client.deleted_collections
    last_upsert = max(
        index for index, event in enumerate(events)
        if event == "qdrant-upsert:" + staging_collection
    )
    assert last_upsert < events.index(
        "postgres-ticket-lineage-switch"
    )
    assert events.index("postgres-ticket-lineage-switch") < events.index(
        "postgres-pointer-switch"
    )


def test_reindex_failure_leaves_active_pointer_and_ticket_lineage_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    client = FakeQdrantAndMlClient(events, "new-embedder", fail_on_embedding_call=2)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: client)
    pool = FakePool(events, add_ticket_during_build=True)

    with pytest.raises(RuntimeError, match="different model version"):
        asyncio.run(
            reindex_qdrant(
                pool,
                {
                    "embedder_version": "new-embedder",
                    "embedding_dimension": 8,
                    "collection_base": "pulse109_new_embedder",
                },
                job_id=43,
            )
        )

    assert "qdrant-upsert:pulse109_new_embedder_job_43" in events
    assert not any(event.startswith("postgres-") for event in events)
    assert pool.active_index["collection_name"] == "active-old"
    assert not client.deleted_collections


def test_reindex_rejects_ml_registry_mismatch_before_creating_staging_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    client = FakeQdrantAndMlClient(
        events,
        "new-embedder",
        metadata_dimension=16,
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: client)
    pool = FakePool(events)

    with pytest.raises(RuntimeError, match="metadata does not match"):
        asyncio.run(
            reindex_qdrant(
                pool,
                {
                    "embedder_version": "new-embedder",
                    "embedding_dimension": 8,
                    "collection_base": "pulse109_new_embedder",
                },
                job_id=44,
            )
        )

    assert not client.collections
    assert not any(event.startswith("postgres-") for event in events)
