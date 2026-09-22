"""OpenAI-compatible embeddings adapter used by semantic memory."""

from collections.abc import Sequence

import httpx

from core.config import EmbeddingConfig


class OpenAICompatibleEmbeddingProvider:
    def __init__(self, settings: EmbeddingConfig):
        if not settings.base_url or not settings.model:
            raise ValueError("Embedding base_url and model must be configured")
        self._settings = settings

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_many([text])
        return vectors[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self._settings.api_key:
            headers["Authorization"] = f"Bearer {self._settings.api_key}"
        async with httpx.AsyncClient(timeout=self._settings.timeout) as client:
            response = await client.post(
                self._endpoint(),
                headers=headers,
                json={"model": self._settings.model, "input": texts},
            )
            response.raise_for_status()
        payload = response.json()
        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ValueError("Embedding provider returned an unexpected data shape")
        vectors: list[list[float]] = []
        for row in rows:
            vector = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
                raise ValueError("Embedding provider returned a non-vector embedding")
            values = [float(value) for value in vector]
            if self._settings.dimension and len(values) != self._settings.dimension:
                raise ValueError(
                    "Embedding dimension does not match EMBEDDING_DIMENSION "
                    f"({len(values)} != {self._settings.dimension})"
                )
            vectors.append(values)
        return vectors

    def _endpoint(self) -> str:
        base_url = self._settings.base_url.rstrip("/")
        return base_url if base_url.endswith("/embeddings") else f"{base_url}/embeddings"
