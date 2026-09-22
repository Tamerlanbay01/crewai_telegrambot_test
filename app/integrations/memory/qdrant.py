"""Qdrant implementation of the semantic-memory search index."""

from datetime import datetime
from uuid import UUID

from core.config import QdrantConfig
from integrations.memory.index import MemorySearchHit
from models.memory import Memory, MemoryScope


class QdrantMemorySearchIndex:
    """A non-authoritative vector index; canonical content stays in PostgreSQL."""

    def __init__(self, settings: QdrantConfig, *, dimension: int):
        if not settings.url:
            raise ValueError("QDRANT_URL must be configured")
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        try:
            from qdrant_client import AsyncQdrantClient, models
        except ImportError as exc:  # pragma: no cover - depends on optional production package
            raise RuntimeError("Install qdrant-client to enable semantic memory") from exc
        self._models = models
        self._settings = settings
        self._dimension = dimension
        self._client = AsyncQdrantClient(
            url=settings.url,
            api_key=settings.api_key or None,
            timeout=settings.timeout,
        )

    async def ensure_collection(self) -> None:
        if not await self._client.collection_exists(self._settings.collection_name):
            await self._client.create_collection(
                collection_name=self._settings.collection_name,
                vectors_config=self._models.VectorParams(
                    size=self._dimension,
                    distance=self._models.Distance.COSINE,
                ),
            )
        for field, schema in (
            ("user_id", self._models.PayloadSchemaType.INTEGER),
            ("scope", self._models.PayloadSchemaType.KEYWORD),
            ("memory_type", self._models.PayloadSchemaType.KEYWORD),
            ("agent_id", self._models.PayloadSchemaType.KEYWORD),
            ("run_id", self._models.PayloadSchemaType.KEYWORD),
        ):
            await self._client.create_payload_index(
                collection_name=self._settings.collection_name,
                field_name=field,
                field_schema=schema,
                wait=True,
            )

    async def upsert(self, memory: Memory, vector: list[float]) -> None:
        await self._client.upsert(
            collection_name=self._settings.collection_name,
            points=[
                self._models.PointStruct(
                    id=str(memory.id),
                    vector=vector,
                    payload=self._payload(memory),
                )
            ],
            wait=True,
        )

    async def delete(self, memory_id: UUID) -> None:
        await self._client.delete(
            collection_name=self._settings.collection_name,
            points_selector=self._models.PointIdsList(points=[str(memory_id)]),
            wait=True,
        )

    async def search(
        self,
        vector: list[float],
        *,
        user_id: int,
        scopes: list[MemoryScope],
        agent_id: UUID | None,
        limit: int,
        now: datetime,
    ) -> list[MemorySearchHit]:
        response = await self._client.query_points(
            collection_name=self._settings.collection_name,
            query=vector,
            query_filter=self._filter(
                user_id=user_id,
                scopes=scopes,
                agent_id=agent_id,
                now=now,
            ),
            limit=limit,
            with_payload=["memory_id"],
            with_vectors=False,
        )
        hits: list[MemorySearchHit] = []
        for point in response.points:
            memory_id = point.payload.get("memory_id") if point.payload else None
            if not memory_id:
                continue
            try:
                hits.append(MemorySearchHit(memory_id=UUID(str(memory_id)), score=point.score))
            except (TypeError, ValueError):
                continue
        return hits

    def _filter(
        self,
        *,
        user_id: int,
        scopes: list[MemoryScope],
        agent_id: UUID | None,
        now: datetime,
    ):
        scope_conditions = []
        if MemoryScope.USER_GLOBAL in scopes:
            scope_conditions.append(
                self._models.FieldCondition(
                    key="scope", match=self._models.MatchValue(value=MemoryScope.USER_GLOBAL.value)
                )
            )
        if MemoryScope.AGENT_PRIVATE in scopes and agent_id is not None:
            scope_conditions.append(
                self._models.Filter(
                    must=[
                        self._models.FieldCondition(
                            key="scope",
                            match=self._models.MatchValue(value=MemoryScope.AGENT_PRIVATE.value),
                        ),
                        self._models.FieldCondition(
                            key="agent_id",
                            match=self._models.MatchValue(value=str(agent_id)),
                        ),
                    ]
                )
            )
        if not scope_conditions:
            return self._models.Filter(must=[self._models.HasIdCondition(has_id=[])])
        active_or_unexpired = self._models.Filter(
            should=[
                self._models.IsNullCondition(
                    is_null=self._models.PayloadField(key="expires_at")
                ),
                self._models.FieldCondition(
                    key="expires_at",
                    range=self._models.DatetimeRange(gt=now.isoformat()),
                ),
            ],
            min_should=self._models.MinShould(
                conditions=[
                    self._models.IsNullCondition(
                        is_null=self._models.PayloadField(key="expires_at")
                    ),
                    self._models.FieldCondition(
                        key="expires_at",
                        range=self._models.DatetimeRange(gt=now.isoformat()),
                    ),
                ],
                min_count=1,
            ),
        )
        return self._models.Filter(
            must=[
                self._models.FieldCondition(
                    key="user_id", match=self._models.MatchValue(value=user_id)
                ),
                self._models.Filter(
                    should=scope_conditions,
                    min_should=self._models.MinShould(
                        conditions=scope_conditions, min_count=1
                    ),
                ),
                active_or_unexpired,
            ]
        )

    @staticmethod
    def _payload(memory: Memory) -> dict[str, object]:
        return {
            "memory_id": str(memory.id),
            "user_id": memory.user_id,
            "scope": memory.scope.value,
            "memory_type": memory.memory_type.value,
            "agent_id": str(memory.agent_id) if memory.agent_id else None,
            "run_id": str(memory.run_id) if memory.run_id else None,
            "key": memory.key,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
        }
